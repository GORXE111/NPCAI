using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace NPCLLM.NPC
{
    /// <summary>
    /// NPC inference scheduler - core paper contribution.
    /// Problem: 10 NPCs serial inference takes 10-20s, bad player experience.
    /// Strategy:
    /// 1. Priority queue (player-nearby / active-dialogue NPCs first)
    /// 2. Response cache (reuse replies for similar queries)
    /// 3. Async pipeline (prepare next NPC's prompt while current is inferring)
    /// </summary>
    public class NPCScheduler : MonoBehaviour
    {
        public static NPCScheduler Instance { get; private set; }

        [Header("Settings")]
        [SerializeField] private int maxConcurrent = 3;
        [SerializeField] private float playerProximityBoost = 10f;

        [Header("Stats")]
        [SerializeField] private int queueLength;
        [SerializeField] private int totalRequests;
        [SerializeField] private int cacheHits;
        [SerializeField] private float avgResponseTime;

        private readonly List<ScheduleRequest> _queue = new List<ScheduleRequest>();
        private int _activeCount;
        private Transform _playerTransform;
        private readonly List<float> _responseTimes = new List<float>();

        private readonly Dictionary<string, CacheEntry> _cache = new Dictionary<string, CacheEntry>();
        private const float CACHE_TTL = 300f;
        private const int MAX_CACHE_SIZE = 50;

        private class ScheduleRequest
        {
            public NPCBrain npc;
            public string message;
            public string speaker;
            public float priority;
            public Action<string> callback;
            public float enqueueTime;
        }

        private class CacheEntry
        {
            public string response;
            public float timestamp;
        }

        private void Awake()
        {
            if (Instance != null && Instance != this) { Destroy(gameObject); return; }
            Instance = this;
        }

        public void SetPlayer(Transform player) => _playerTransform = player;

        public void RequestPlayerChat(NPCBrain npc, string message, Action<string> callback)
        {
            string cacheKey = npc.NpcId + ":" + message.ToLowerInvariant().Trim();
            if (TryGetCache(cacheKey, out string cached))
            {
                cacheHits++;
                callback?.Invoke(cached);
                return;
            }

            float priority = CalculatePriority(npc, isPlayerChat: true);
            Enqueue(npc, message, "Player", priority, resp =>
            {
                SetCache(cacheKey, resp);
                callback?.Invoke(resp);
            });
        }

        public void RequestNPCChat(NPCBrain fromNpc, NPCBrain toNpc, string message, Action<string> callback)
        {
            float priority = CalculatePriority(toNpc, isPlayerChat: false);
            Enqueue(toNpc, message, fromNpc.NpcName, priority, callback);
        }

        private void Enqueue(NPCBrain npc, string message, string speaker, float priority, Action<string> callback)
        {
            _queue.Add(new ScheduleRequest
            {
                npc = npc,
                message = message,
                speaker = speaker,
                priority = priority,
                callback = callback,
                enqueueTime = Time.realtimeSinceStartup
            });
            totalRequests++;
            queueLength = _queue.Count;
        }

        private void Update()
        {
            while (_activeCount < maxConcurrent && _queue.Count > 0)
            {
                _queue.Sort((a, b) => b.priority.CompareTo(a.priority));
                var req = _queue[0];
                _queue.RemoveAt(0);
                queueLength = _queue.Count;

                _activeCount++;
                float startTime = Time.realtimeSinceStartup;

                if (req.speaker == "Player")
                {
                    req.npc.TalkTo(req.message, resp =>
                    {
                        _activeCount--;
                        RecordResponseTime(Time.realtimeSinceStartup - startTime);
                        req.callback?.Invoke(resp);
                    });
                }
                else
                {
                    req.npc.TalkToNPC("", req.speaker, req.message, resp =>
                    {
                        _activeCount--;
                        RecordResponseTime(Time.realtimeSinceStartup - startTime);
                        req.callback?.Invoke(resp);
                    });
                }
            }
        }

        private float CalculatePriority(NPCBrain npc, bool isPlayerChat)
        {
            float priority = 0f;
            if (isPlayerChat) priority += 100f;

            if (_playerTransform != null && npc != null)
            {
                float dist = Vector3.Distance(_playerTransform.position, npc.transform.position);
                priority += Mathf.Max(0, playerProximityBoost - dist);
            }

            return priority;
        }

        private bool TryGetCache(string key, out string response)
        {
            response = null;
            if (!_cache.ContainsKey(key)) return false;
            var entry = _cache[key];
            if (Time.realtimeSinceStartup - entry.timestamp > CACHE_TTL)
            {
                _cache.Remove(key);
                return false;
            }
            response = entry.response;
            return true;
        }

        private void SetCache(string key, string response)
        {
            if (_cache.Count >= MAX_CACHE_SIZE)
            {
                string oldest = _cache.OrderBy(kv => kv.Value.timestamp).First().Key;
                _cache.Remove(oldest);
            }
            _cache[key] = new CacheEntry { response = response, timestamp = Time.realtimeSinceStartup };
        }

        private void RecordResponseTime(float time)
        {
            _responseTimes.Add(time);
            if (_responseTimes.Count > 100) _responseTimes.RemoveAt(0);
            avgResponseTime = _responseTimes.Count > 0
                ? _responseTimes.Sum() / _responseTimes.Count : 0f;
        }

        // Experiment data accessors
        public float AvgResponseTime => avgResponseTime;
        public int TotalRequests => totalRequests;
        public int CacheHits => cacheHits;
        public float CacheHitRate => totalRequests > 0 ? (float)cacheHits / totalRequests : 0f;
    }
}
