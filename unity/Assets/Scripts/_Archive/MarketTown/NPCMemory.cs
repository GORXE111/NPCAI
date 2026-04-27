using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace NPCLLM.NPC
{
    [Serializable]
    public class MemoryEntry
    {
        public string id;
        public string content;
        public string category;  // episodic, semantic, social
        public float importance;  // 0-1
        public string timestamp;
        public int accessCount;
        public string lastAccess;
        public float strength;    // memory strength, decays over time

        public MemoryEntry(string content, string category, float importance)
        {
            id = Guid.NewGuid().ToString("N").Substring(0, 8);
            this.content = content;
            this.category = category;
            this.importance = importance;
            timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            lastAccess = timestamp;
            accessCount = 0;
            strength = 1f;
        }
    }

    [Serializable]
    public class MemoryStore
    {
        public List<MemoryEntry> entries = new List<MemoryEntry>();
    }

    /// <summary>
    /// Four-layer memory system + Ebbinghaus forgetting curve
    /// - ShortTerm: current dialogue context (in memory, not persisted)
    /// - Episodic: specific events ("Player bought a sword at 13:00")
    /// - Semantic: world knowledge ("The smithy is east of the square")
    /// - Social: managed by SocialGraph
    /// </summary>
    public class NPCMemory
    {
        private readonly string _npcId;
        private readonly string _savePath;

        private readonly List<string> _shortTerm = new List<string>();
        private const int SHORT_TERM_CAPACITY = 10;

        private MemoryStore _longTerm = new MemoryStore();

        private const float DECAY_RATE = 0.15f;
        private const float FORGET_THRESHOLD = 0.1f;

        public NPCMemory(string npcId, string savePath)
        {
            _npcId = npcId;
            _savePath = Path.Combine(savePath, npcId + "_memory.json");
            Load();
        }

        public void AddShortTerm(string content)
        {
            _shortTerm.Add(content);
            if (_shortTerm.Count > SHORT_TERM_CAPACITY)
                _shortTerm.RemoveAt(0);
        }

        public List<string> GetShortTerm() => new List<string>(_shortTerm);

        public void ClearShortTerm() => _shortTerm.Clear();

        public void AddMemory(string content, string category, float importance)
        {
            var entry = new MemoryEntry(content, category, importance);
            _longTerm.entries.Add(entry);
            Save();
        }

        public List<MemoryEntry> Recall(string category = null, int limit = 5)
        {
            ApplyForgetting();

            var query = _longTerm.entries.AsEnumerable();
            if (category != null)
                query = query.Where(e => e.category == category);

            var results = query
                .OrderByDescending(e => e.strength * e.importance)
                .Take(limit)
                .ToList();

            string now = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            foreach (var entry in results)
            {
                entry.accessCount++;
                entry.lastAccess = now;
                entry.strength = Mathf.Min(1f, entry.strength + 0.1f);
            }

            Save();
            return results;
        }

        public string GetMemorySummary(int maxEntries = 5)
        {
            var memories = Recall(limit: maxEntries);
            if (memories.Count == 0) return "No significant memories.";

            var lines = new List<string>();
            foreach (var m in memories)
                lines.Add("[" + m.category + "] " + m.content);
            return string.Join("\n", lines);
        }

        private void ApplyForgetting()
        {
            var toRemove = new List<MemoryEntry>();

            foreach (var entry in _longTerm.entries)
            {
                if (DateTime.TryParse(entry.lastAccess, out DateTime lastAccess))
                {
                    double hoursSince = (DateTime.Now - lastAccess).TotalHours;
                    float stability = (1f + entry.accessCount * 0.5f) * (0.5f + entry.importance * 0.5f);
                    entry.strength = Mathf.Exp(-DECAY_RATE * (float)hoursSince / stability);
                }

                if (entry.strength < FORGET_THRESHOLD)
                    toRemove.Add(entry);
            }

            foreach (var entry in toRemove)
                _longTerm.entries.Remove(entry);
        }

        public int MemoryCount => _longTerm.entries.Count;

        public void Save()
        {
            try
            {
                string dir = Path.GetDirectoryName(_savePath);
                if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
                string json = JsonUtility.ToJson(_longTerm, true);
                File.WriteAllText(_savePath, json);
            }
            catch (Exception e)
            {
                Debug.LogWarning("[NPCMemory] Save failed for " + _npcId + ": " + e.Message);
            }
        }

        public void Load()
        {
            try
            {
                if (File.Exists(_savePath))
                {
                    string json = File.ReadAllText(_savePath);
                    _longTerm = JsonUtility.FromJson<MemoryStore>(json);
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning("[NPCMemory] Load failed for " + _npcId + ": " + e.Message);
                _longTerm = new MemoryStore();
            }
        }
    }
}
