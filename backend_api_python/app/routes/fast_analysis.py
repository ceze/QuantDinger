"""
Fast Analysis API Routes

New high-performance analysis endpoints that replace the slow multi-agent system.
"""
from flask import g, jsonify, request
from app.openapi.blueprint import HumanBlueprint as Blueprint
import threading
import time

from app.utils.auth import login_required
from app.utils.logger import get_logger
from app.services.fast_analysis import get_fast_analysis_service
from app.services.analysis_memory import get_analysis_memory
from app.services.billing_service import get_billing_service
from flask_cors import cross_origin

logger = get_logger(__name__)

fast_analysis_blp = Blueprint('fast_analysis', __name__)

# L1 Result cache (in-process dict, fastest): share analysis result among concurrent requests
# Key: "market:symbol:timeframe:language" -> {"result": ..., "timestamp": ...}
_analysis_result_cache = {}
_analysis_cache_lock = threading.Lock()
_ANALYSIS_CACHE_TTL = 5*60  # 分析缓存时间（秒）

# L2 Redis cache key prefix (cross-process sharing for multi-instance deployments)
_REDIS_CACHE_PREFIX = "fa_cache:"

# Lazy-loaded CacheManager (Redis or fallback MemoryCache)
_cache_manager = None
_cache_manager_lock = threading.Lock()


def _get_cache_manager():
    """Lazy-load CacheManager. Returns None on any failure (graceful degradation)."""
    global _cache_manager
    if _cache_manager is not None:
        return _cache_manager
    with _cache_manager_lock:
        if _cache_manager is not None:
            return _cache_manager
        try:
            from app.utils.cache import CacheManager
            _cache_manager = CacheManager()
            backend = "Redis" if _cache_manager.is_redis else "MemoryCache"
            logger.info(f"Fast analysis L2 cache backend: {backend}")
            return _cache_manager
        except Exception as e:
            logger.info(f"L2 cache unavailable, using L1 memory only: {e}")
            return None


def _cache_get(cache_key: str):
    """
    Two-tier cache lookup: L1 (in-process dict) -> L2 (Redis/CacheManager).
    Returns cached result dict on hit, None on miss.
    On L2 hit, backfills L1 for subsequent fast access.
    """
    # L1: in-process dict (fastest)
    with _analysis_cache_lock:
        if cache_key in _analysis_result_cache:
            cached = _analysis_result_cache[cache_key]
            if time.time() - cached['timestamp'] < _ANALYSIS_CACHE_TTL:
                return cached['result'], 'memory'
            else:
                del _analysis_result_cache[cache_key]

    # L2: Redis / CacheManager
    cm = _get_cache_manager()
    if cm is not None:
        try:
            redis_key = f"{_REDIS_CACHE_PREFIX}{cache_key}"
            data = cm.get(redis_key)
            if data and isinstance(data, dict):
                result = data.get('result')
                ts = data.get('timestamp', 0)
                if result and time.time() - ts < _ANALYSIS_CACHE_TTL:
                    # Backfill L1 for subsequent fast access
                    with _analysis_cache_lock:
                        _analysis_result_cache[cache_key] = {
                            'result': result,
                            'timestamp': ts
                        }
                    return result, 'redis'
        except Exception as e:
            logger.debug(f"L2 cache read failed for {cache_key}: {e}")

    return None, None


def _cache_set(cache_key: str, result: dict):
    """
    Write to both L1 (in-process dict) and L2 (Redis/CacheManager).
    L2 failure is silently ignored (graceful degradation).
    """
    ts = time.time()
    # L1: always write
    with _analysis_cache_lock:
        _analysis_result_cache[cache_key] = {
            'result': result,
            'timestamp': ts
        }

    # L2: best-effort write to Redis
    cm = _get_cache_manager()
    if cm is not None:
        try:
            redis_key = f"{_REDIS_CACHE_PREFIX}{cache_key}"
            cm.set(redis_key, {'result': result, 'timestamp': ts}, ttl=_ANALYSIS_CACHE_TTL)
            logger.info(f"Cached result to L1+L2 for {cache_key}")
            return
        except Exception as e:
            logger.debug(f"L2 cache write failed for {cache_key}: {e}")

    logger.info(f"Cached result to L1 only for {cache_key}")


def _try_refund_credits(user_id: int, amount: int, remark: str):
    """Best-effort async refund when task fails after pre-charge."""
    try:
        if int(amount or 0) <= 0:
            return
        billing = get_billing_service()
        billing.add_credits(
            user_id=int(user_id),
            amount=int(amount),
            action='refund',
            remark=remark
        )
    except Exception as e:
        logger.error(f"Async auto refund failed: {e}", exc_info=True)


def _build_cache_key(market: str, symbol: str, timeframe: str, language: str) -> str:
    return f"{(market or '').strip().upper()}:{(symbol or '').strip().upper()}:{(timeframe or '1D').strip().upper()}:{language}"


def _db_get_recent_result(market: str, symbol: str, timeframe: str, max_age_minutes: int = 5) -> dict | None:
    """Check DB for a recently completed or pending analysis.

    This is the ultimate safety net against duplicate task creation — even when
    L1/L2 caches miss (multi-worker gunicorn, Redis unavailable), the DB
    always has the truth.

    Returns a dict with at least ``{"task_id", "task_status"}`` and, for
    completed tasks, the full result payload.
    """
    try:
        from app.utils.db import get_db_connection
        with get_db_connection() as db:
            cur = db.cursor()
            cur.execute("""
                SELECT id, task_status, decision, confidence, price_at_analysis,
                       summary, reasons, scores,
                       created_at, timeframe, raw_result
                FROM qd_analysis_memory
                WHERE market = %s AND symbol = %s AND (timeframe = %s OR timeframe IS NULL)
                  AND task_status IN ('pending', 'processing', 'completed')
                  AND created_at > NOW() - (%s || ' minutes')::interval
                ORDER BY created_at DESC
                LIMIT 1
            """, (market.strip(), symbol.strip(), (timeframe or '1D').strip(), int(max_age_minutes)))
            row = cur.fetchone()
            if not row:
                return None

            task_id = row['id']
            task_status = row['task_status']

            # Pending/processing task — return task_id so the caller can return "submitted"
            if task_status in ('pending', 'processing'):
                return {
                    'task_id': task_id,
                    'memory_id': task_id,
                    'task_status': 'pending',
                }

            # Completed task — return full result
            raw = row['raw_result']
            if raw and isinstance(raw, dict) and raw.get('decision'):
                raw['_db_cached'] = True
                raw['_memory_id'] = task_id
                raw['task_status'] = 'completed'
                return raw
            # Fallback: reconstruct from columns
            return {
                'decision': row['decision'],
                'confidence': row['confidence'],
                'price_at_analysis': float(row['price_at_analysis']) if row['price_at_analysis'] else None,
                'summary': row['summary'],
                'reasons': row['reasons'],
                'scores': row['scores'],
                'memory_id': task_id,
                'task_status': 'completed',
                '_db_cached': True,
            }
    except Exception as e:
        logger.warning(f"DB recent-result check failed: {e}", exc_info=True)
        return None


def _run_async_analysis_task(task_memory_id: int, market: str, symbol: str, language: str,
                             model: str, timeframe: str, user_id: int,
                             credits_charged: int = 0):
    """
    Background worker: execute analysis and update pending history record.
    """
    logger.info(f"Async task started: memory_id={task_memory_id}, {market}:{symbol}")
    try:
        service = get_fast_analysis_service()
        memory = get_analysis_memory()
        result = service.analyze(
            market=market,
            symbol=symbol,
            language=language,
            model=model,
            timeframe=timeframe,
            user_id=user_id
        )
        logger.info(f"Async task analysis completed: memory_id={task_memory_id}, decision={result.get('decision')}")
        memory.finalize_pending_task(task_memory_id, result)
        logger.info(f"Async task finalized: memory_id={task_memory_id}, status={ 'completed' if not result.get('error') else 'failed'}")
        if result.get("error"):
            _try_refund_credits(
                user_id=int(user_id),
                amount=int(credits_charged or 0),
                remark=f'Auto refund: async fast-analysis failed ({market}:{symbol}:{timeframe})'
            )
        else:
            # Cache successful result to L1+L2 for subsequent requests
            cache_key = _build_cache_key(market, symbol, timeframe, language)
            _cache_set(cache_key, result)

        # analyze() already stores a separate memory row; remove it to avoid duplicates.
        auto_memory_id = result.get("memory_id")
        if auto_memory_id and int(auto_memory_id) != int(task_memory_id):
            try:
                memory.delete_history(int(auto_memory_id), user_id=user_id)
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Async analysis task failed: {e}", exc_info=True)
        _try_refund_credits(
            user_id=int(user_id),
            amount=int(credits_charged or 0),
            remark=f'Auto refund: async fast-analysis exception ({market}:{symbol}:{timeframe})'
        )
        try:
            get_analysis_memory().fail_pending_task(task_memory_id, str(e))
            logger.error(f"Async task marked as failed: memory_id={task_memory_id}")
        except Exception:
            pass


@fast_analysis_blp.route('/analyze', methods=['POST'])
# @login_required  # Disabled: allow anonymous access
@cross_origin()  # Allow CORS for this route
def analyze():
    """
    Fast AI analysis for any symbol.

    Request body:
        market (required): Crypto, USStock, Forex, etc.
        symbol (required): e.g. BTC/USDT, AAPL
        language (optional, default en-US): Response language
        model (optional): LLM model id, e.g. openai/gpt-4o
        timeframe (optional, default 1D): Analysis timeframe
        async_submit (optional): Submit as background task
    """
    try:
        data = request.get_json(silent=True) or dict(request.form)
        
        market = (data.get('market') or '').strip()
        symbol = (data.get('symbol') or '').strip()
        language = data.get('language', 'en-US')
        model = data.get('model')
        timeframe = data.get('timeframe', '1D')
        async_submit = bool(data.get('async_submit', False))
        
        if not market or not symbol:
            return jsonify({
                'code': 0,
                'msg': 'market and symbol are required',
                'data': None
            }), 400
        
        # All requests use anonymous user_id=88888
        user_id = 88888
        logger.info(f"Analysis request (user_id={user_id}): {market}:{symbol}")
        
        # Build cache key (independent of user_id - shared across all users)
        cache_key = _build_cache_key(market, symbol, timeframe, language)
        
        # Check result cache: L1 (memory) -> L2 (Redis)
        cached_result, cache_source = _cache_get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit ({cache_source}) for {cache_key}, returning cached result")
            return jsonify({
                'code': 1,
                'msg': 'success (cached)',
                'data': {
                    **cached_result,
                    'market': market,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'credits_charged': 0,
                    'remaining_credits': None,
                    '_cached': True
                }
            })
        logger.info(f"Cache miss (L1+L2) for {cache_key}, checking DB for recent result")

        # L3: Database fallback — find a recently completed analysis for the same
        # market/symbol/timeframe.  This guards against duplicate task creation
        # when L1 (process-local) and L2 (Redis) caches miss due to multi-worker
        # gunicorn deployments or transient Redis failures.
        db_result = _db_get_recent_result(market, symbol, timeframe, max_age_minutes=5)
        if db_result is not None:
            task_status = db_result.get('task_status', 'completed')
            db_task_id = db_result.get('task_id') or db_result.get('_memory_id')

            if task_status == 'pending':
                # A pending task exists — return "submitted" instead of creating a duplicate
                logger.info(f"DB pending task found for {cache_key}, returning submitted (task_id={db_task_id})")
                return jsonify({
                    'code': 1,
                    'msg': 'submitted',
                    'data': {
                        'task_id': db_task_id,
                        'memory_id': db_task_id,
                        'status': 'processing',
                        'market': market,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'credits_charged': 0,
                        'remaining_credits': None,
                    }
                })

            # Completed task — return cached result
            logger.info(f"DB cache hit for {cache_key}, returning recent result (memory_id={db_task_id})")
            return jsonify({
                'code': 1,
                'msg': 'success (cached)',
                'data': {
                    **db_result,
                    'market': market,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'credits_charged': 0,
                    'remaining_credits': None,
                }
            })
            
        # Billing / credits (best-effort) - Anonymous users (88888) skip billing
        credits_charged = 0
        remaining_credits = None
        billing_consumed = False
        billing = None
        try:
            if user_id and int(user_id) != 88888:  # Only charge authenticated users
                billing = get_billing_service()
                if billing.is_billing_enabled():
                    credits_charged = int(billing.get_feature_cost('ai_analysis') or 0)
                    if credits_charged > 0:
                        ok, msg = billing.check_and_consume(
                            user_id=int(user_id),
                            feature='ai_analysis',
                            reference_id=f"fast_analysis_{market}:{symbol}:{timeframe}"
                        )
                        if not ok:
                            # Standardize insufficient credits message
                            if str(msg or "").startswith('insufficient_credits'):
                                # Format: insufficient_credits:<current>:<cost>
                                parts = str(msg).split(':')
                                cur = float(parts[1]) if len(parts) >= 2 else 0.0
                                req = float(parts[2]) if len(parts) >= 3 else float(credits_charged)
                                return jsonify({
                                    'code': 0,
                                    'msg': 'Insufficient credits',
                                    'data': {
                                        'required': req,
                                        'current': cur,
                                        'shortage': max(0.0, req - cur),
                                    }
                                }), 400
                            return jsonify({'code': 0, 'msg': f'Failed to deduct credits: {msg}', 'data': None}), 500
                        billing_consumed = True
                        # Query remaining credits after successful consumption
                        try:
                            remaining_credits = float(billing.get_user_credits(int(user_id)))
                        except Exception:
                            remaining_credits = None
        except Exception as e:
            # Billing failure should not crash analysis by default, but should be visible in logs.
            logger.warning(f"Billing check failed (skipped): {e}", exc_info=True)
        
        service = get_fast_analysis_service()

        # Async submit mode: support anonymous users
        if async_submit:
            # Check cache: L1 (memory) -> L2 (Redis) before creating async task
            cached_result, cache_source = _cache_get(cache_key)
            if cached_result is not None:
                logger.info(f"Async submit cache hit ({cache_source}) for {cache_key}")
                return jsonify({
                    'code': 1,
                    'msg': 'success (cached)',
                    'data': {
                        **cached_result,
                        'market': market,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'credits_charged': 0,
                        'remaining_credits': None,
                        '_cached': True
                    }
                })
            logger.info(f"Async submit cache miss (L1+L2) for {cache_key}")
            
            # Anonymous users can use async mode with user_id=88888
            memory = get_analysis_memory()
            pending_id = memory.create_pending_task(
                market=market,
                symbol=symbol,
                language=language,
                model=model or "",
                timeframe=timeframe,
                user_id=user_id
            )
            if not pending_id:
                return jsonify({'code': 0, 'msg': 'Failed to create analysis task', 'data': None}), 500

            t = threading.Thread(
                target=_run_async_analysis_task,
                args=(int(pending_id), market, symbol, language, model, timeframe, int(user_id), int(credits_charged or 0)),
                daemon=True
            )
            t.start()

            return jsonify({
                'code': 1,
                'msg': 'submitted',
                'data': {
                    'task_id': int(pending_id),
                    'memory_id': int(pending_id),
                    'status': 'processing',
                    'market': market,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'credits_charged': credits_charged,
                    'remaining_credits': remaining_credits,
                }
            })

        result = service.analyze(
            market=market,
            symbol=symbol,
            language=language,
            model=model,
            timeframe=timeframe,
            user_id=user_id
        )
        
        if result.get('error'):
            # Best-effort refund if we already charged but analysis failed.
            if billing_consumed and billing and credits_charged > 0:
                try:
                    billing.add_credits(
                        user_id=int(user_id),
                        amount=int(credits_charged),
                        action='refund',
                        remark=f'Auto refund: fast-analysis failed ({market}:{symbol}:{timeframe})'
                    )
                    remaining_credits = float(billing.get_user_credits(int(user_id)))
                except Exception as re:
                    logger.error(f"Auto refund failed: {re}", exc_info=True)
            return jsonify({
                'code': 0,
                'msg': result['error'],
                'data': result
            }), 500
        
        # Cache the result to L1+L2 for other concurrent requests
        _cache_set(cache_key, result)
        
        # memory_id is already set in service.analyze() -> _store_analysis_memory()
        # No need to store again here (would create duplicates)
        
        return jsonify({
            'code': 1,
            'msg': 'success',
            'data': {
                **(result or {}),
                'market': market,
                'symbol': symbol,
                'timeframe': timeframe,
                'credits_charged': credits_charged,
                'remaining_credits': remaining_credits,
            }
        })
        
    except Exception as e:
        # Best-effort refund on unexpected error after charge.
        try:
            if 'billing_consumed' in locals() and billing_consumed and 'billing' in locals() and billing and credits_charged > 0 and 'user_id' in locals() and user_id:
                billing.add_credits(
                    user_id=int(user_id),
                    amount=int(credits_charged),
                    action='refund',
                    remark=f'Auto refund: fast-analysis exception ({market}:{symbol}:{timeframe})'
                )
        except Exception:
            pass
        logger.error(f"Fast analysis API failed: {e}", exc_info=True)
        return jsonify({
            'code': 0,
            'msg': str(e),
            'data': None
        }), 500


@fast_analysis_blp.route('/analyze-legacy', methods=['POST'])
@login_required
def analyze_legacy():
    """
    Fast analysis with legacy format output.
    For backward compatibility with existing frontend.
    
    POST /api/fast-analysis/analyze-legacy
    Body: Same as /analyze
    
    Returns:
        Result in multi-agent format for frontend compatibility.
    """
    try:
        data = request.get_json(silent=True) or dict(request.form)
        
        market = (data.get('market') or '').strip()
        symbol = (data.get('symbol') or '').strip()
        language = data.get('language', 'en-US')
        model = data.get('model')
        timeframe = data.get('timeframe', '1D')
        
        if not market or not symbol:
            return jsonify({
                'code': 0,
                'msg': 'market and symbol are required',
                'data': None
            }), 400

        # Billing / credits (same behavior as /analyze)
        user_id = getattr(g, 'user_id', None)
        if not user_id:
            return jsonify({'code': 0, 'msg': 'Unauthorized', 'data': None}), 401

        credits_charged = 0
        remaining_credits = None
        billing_consumed = False
        billing = None
        try:
            billing = get_billing_service()
            if billing.is_billing_enabled():
                credits_charged = int(billing.get_feature_cost('ai_analysis') or 0)
                if credits_charged > 0:
                    ok, msg = billing.check_and_consume(
                        user_id=int(user_id),
                        feature='ai_analysis',
                        reference_id=f"fast_analysis_legacy_{market}:{symbol}:{timeframe}"
                    )
                    if not ok:
                        if str(msg or "").startswith('insufficient_credits'):
                            parts = str(msg).split(':')
                            cur = float(parts[1]) if len(parts) >= 2 else 0.0
                            req = float(parts[2]) if len(parts) >= 3 else float(credits_charged)
                            return jsonify({
                                'code': 0,
                                'msg': 'Insufficient credits',
                                'data': {
                                    'required': req,
                                    'current': cur,
                                    'shortage': max(0.0, req - cur),
                                }
                            }), 400
                        return jsonify({'code': 0, 'msg': f'Failed to deduct credits: {msg}', 'data': None}), 500
                    billing_consumed = True
                    try:
                        remaining_credits = float(billing.get_user_credits(int(user_id)))
                    except Exception:
                        remaining_credits = None
        except Exception as e:
            logger.warning(f"Billing check failed (skipped): {e}", exc_info=True)
        
        service = get_fast_analysis_service()
        result = service.analyze_legacy_format(
            market=market,
            symbol=symbol,
            language=language,
            model=model,
            timeframe=timeframe
        )
        
        if result.get('error'):
            if billing_consumed and billing and credits_charged > 0:
                try:
                    billing.add_credits(
                        user_id=int(user_id),
                        amount=int(credits_charged),
                        action='refund',
                        remark=f'Auto refund: fast-analysis-legacy failed ({market}:{symbol}:{timeframe})'
                    )
                    remaining_credits = float(billing.get_user_credits(int(user_id)))
                except Exception as re:
                    logger.error(f"Auto refund failed (legacy): {re}", exc_info=True)
            return jsonify({
                'code': 0,
                'msg': result['error'],
                'data': result
            }), 500
        
        return jsonify({
            'code': 1,
            'msg': 'success',
            'data': {
                **(result or {}),
                'credits_charged': credits_charged,
                'remaining_credits': remaining_credits,
            }
        })
        
    except Exception as e:
        try:
            if 'billing_consumed' in locals() and billing_consumed and 'billing' in locals() and billing and credits_charged > 0 and 'user_id' in locals() and user_id:
                billing.add_credits(
                    user_id=int(user_id),
                    amount=int(credits_charged),
                    action='refund',
                    remark=f'Auto refund: fast-analysis-legacy exception ({market}:{symbol}:{timeframe})'
                )
        except Exception:
            pass
        logger.error(f"Fast analysis legacy API failed: {e}", exc_info=True)
        return jsonify({
            'code': 0,
            'msg': str(e),
            'data': None
        }), 500


@fast_analysis_blp.route('/history', methods=['GET'])
# @login_required  # Disabled: allow anonymous access
@cross_origin()  # Allow CORS for this route
def get_history():
    """
    Get analysis history for a symbol.
    
    GET /api/fast-analysis/history?market=Crypto&symbol=BTC/USDT&days=7&limit=10
    """
    try:
        memory_id = request.args.get('memory_id', '').strip()
        market = request.args.get('market', '').strip()
        symbol = request.args.get('symbol', '').strip()
        days = int(request.args.get('days', 7))
        limit = min(int(request.args.get('limit', 10)), 50)
        
        memory = get_analysis_memory()
        
        # 如果传入 memory_id，直接返回对应记录
        if memory_id:
            history = memory.get_by_id(memory_id)
            if not history:
                return jsonify({
                    'code': 0,
                    'msg': f'memory_id {memory_id} not found',
                    'data': None
                }), 404
            return jsonify({
                'code': 1,
                'msg': 'success',
                'data': {
                    'items': [history],
                    'total': 1
                }
            })
        
        if not market or not symbol:
            return jsonify({
                'code': 0,
                'msg': 'market and symbol are required',
                'data': None
            }), 400
        
        history = memory.get_recent(market, symbol, days, limit)
        
        return jsonify({
            'code': 1,
            'msg': 'success',
            'data': {
                'items': history,
                'total': len(history)
            }
        })
        
    except Exception as e:
        logger.error(f"Get history failed: {e}")
        return jsonify({
            'code': 0,
            'msg': str(e),
            'data': None
        }), 500


@fast_analysis_blp.route('/history/all', methods=['GET'])
# @login_required  # Disabled: allow anonymous access
@cross_origin()  # Allow CORS for this route
def get_all_history():
    """
    Get all analysis history with pagination.
    Supports both authenticated and anonymous users.
    
    GET /api/fast-analysis/history/all?page=1&pagesize=20
    """
    try:
        page = int(request.args.get('page', 1))
        pagesize = min(int(request.args.get('pagesize', 20)), 50)
        
        # Get current user's ID, fallback to anonymous user_id=88888
        user_id = getattr(g, 'user_id', None)
        if not user_id:
            user_id = 88888  # Anonymous user
        
        memory = get_analysis_memory()
        result = memory.get_all_history(user_id=user_id, page=page, page_size=pagesize)
        
        return jsonify({
            'code': 1,
            'msg': 'success',
            'data': {
                'list': result['items'],
                'total': result['total'],
                'page': result['page'],
                'pagesize': result['page_size']
            }
        })
        
    except Exception as e:
        logger.error(f"Get all history failed: {e}")
        return jsonify({
            'code': 0,
            'msg': str(e),
            'data': None
        }), 500


@fast_analysis_blp.route('/history/<int:memory_id>', methods=['DELETE'])
@login_required
def delete_history(memory_id: int):
    """
    Delete a history record.
    
    DELETE /api/fast-analysis/history/123
    """
    try:
        # Get current user's ID to ensure they can only delete their own records
        user_id = getattr(g, 'user_id', None)
        
        memory = get_analysis_memory()
        success = memory.delete_history(memory_id, user_id=user_id)
        
        if success:
            return jsonify({
                'code': 1,
                'msg': 'Deleted successfully',
                'data': None
            })
        else:
            return jsonify({
                'code': 0,
                'msg': 'Record not found or no permission',
                'data': None
            }), 404
        
    except Exception as e:
        logger.error(f"Delete history failed: {e}")
        return jsonify({
            'code': 0,
            'msg': str(e),
            'data': None
        }), 500


@fast_analysis_blp.route('/feedback', methods=['POST'])
@login_required
def submit_feedback():
    """
    Submit user feedback on an analysis.

    Request body:
        memory_id (required): Analysis history ID
        feedback (required): helpful, not_helpful, accurate, or inaccurate
    """
    try:
        data = request.get_json(silent=True) or dict(request.form)
        
        memory_id = int(data.get('memory_id', 0))
        feedback = (data.get('feedback') or '').strip()
        
        if not memory_id or not feedback:
            return jsonify({
                'code': 0,
                'msg': 'memory_id and feedback are required',
                'data': None
            }), 400
        
        valid_feedback = ['helpful', 'not_helpful', 'accurate', 'inaccurate']
        if feedback not in valid_feedback:
            return jsonify({
                'code': 0,
                'msg': f'feedback must be one of: {valid_feedback}',
                'data': None
            }), 400
        
        memory = get_analysis_memory()
        success = memory.record_feedback(memory_id, feedback)
        
        return jsonify({
            'code': 1 if success else 0,
            'msg': 'success' if success else 'failed',
            'data': None
        })
        
    except Exception as e:
        logger.error(f"Submit feedback failed: {e}")
        return jsonify({
            'code': 0,
            'msg': str(e),
            'data': None
        }), 500


@fast_analysis_blp.route('/performance', methods=['GET'])
# @login_required  # Disabled: allow anonymous access
@cross_origin()  # Allow CORS for this route
def get_performance():
    """
    Get AI analysis performance statistics.
    
    GET /api/fast-analysis/performance?market=Crypto&symbol=BTC/USDT&days=30
    """
    try:
        market = request.args.get('market', '').strip() or None
        symbol = request.args.get('symbol', '').strip() or None
        days = int(request.args.get('days', 30))
        
        memory = get_analysis_memory()
        stats = memory.get_performance_stats(market, symbol, days)
        
        return jsonify({
            'code': 1,
            'msg': 'success',
            'data': stats
        })
        
    except Exception as e:
        logger.error(f"Get performance failed: {e}")
        return jsonify({
            'code': 0,
            'msg': str(e),
            'data': None
        }), 500


@fast_analysis_blp.route('/similar-patterns', methods=['GET'])
@login_required
def get_similar_patterns():
    """
    Get similar historical patterns for current market conditions.
    
    GET /api/fast-analysis/similar-patterns?market=Crypto&symbol=BTC/USDT
    """
    try:
        market = request.args.get('market', '').strip()
        symbol = request.args.get('symbol', '').strip()
        
        if not market or not symbol:
            return jsonify({
                'code': 0,
                'msg': 'market and symbol are required',
                'data': None
            }), 400
        
        # Get current indicators
        service = get_fast_analysis_service()
        data = service._collect_market_data(market, symbol)
        indicators = data.get('indicators', {})
        
        # Find similar patterns
        memory = get_analysis_memory()
        patterns = memory.get_similar_patterns(market, symbol, indicators)
        
        return jsonify({
            'code': 1,
            'msg': 'success',
            'data': {
                'patterns': patterns,
                'current_indicators': {
                    'rsi': indicators.get('rsi', {}).get('value'),
                    'macd_signal': indicators.get('macd', {}).get('signal'),
                    'trend': indicators.get('moving_averages', {}).get('trend'),
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Get similar patterns failed: {e}")
        return jsonify({
            'code': 0,
            'msg': str(e),
            'data': None
        }), 500

# openapi-compat: legacy import name
fast_analysis_bp = fast_analysis_blp
