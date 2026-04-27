using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace NPCLLM.NPC
{
    [Serializable]
    public class SocialRelation
    {
        public string targetId;
        public float affinity;      // -1 to 1 (hostile to close)
        public float trust;         // 0 to 1
        public int interactionCount;
        public string lastInteraction;
        public List<string> sharedMemories;

        public SocialRelation(string targetId)
        {
            this.targetId = targetId;
            affinity = 0f;
            trust = 0.5f;
            interactionCount = 0;
            lastInteraction = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            sharedMemories = new List<string>();
        }
    }

    [Serializable]
    public class SocialData
    {
        public string npcId;
        public List<SocialRelation> relations = new List<SocialRelation>();
    }

    /// <summary>
    /// Social relationship graph - manages NPC relationships, affinity, trust.
    /// Supports information propagation: messages spread along relationship chains, affected by trust.
    /// </summary>
    public class SocialGraph
    {
        private static SocialGraph _instance;
        public static SocialGraph Instance => _instance ??= new SocialGraph();

        private readonly Dictionary<string, SocialData> _graph = new Dictionary<string, SocialData>();
        private string _savePath;

        public void Initialize(string savePath)
        {
            _savePath = Path.Combine(savePath, "social_graph.json");
            Load();
        }

        public void RegisterNPC(string npcId)
        {
            if (!_graph.ContainsKey(npcId))
                _graph[npcId] = new SocialData { npcId = npcId };
        }

        public SocialRelation GetRelation(string fromId, string toId)
        {
            if (!_graph.ContainsKey(fromId)) return null;
            return _graph[fromId].relations.Find(r => r.targetId == toId);
        }

        public SocialRelation GetOrCreateRelation(string fromId, string toId)
        {
            RegisterNPC(fromId);
            var data = _graph[fromId];
            var rel = data.relations.Find(r => r.targetId == toId);
            if (rel == null)
            {
                rel = new SocialRelation(toId);
                data.relations.Add(rel);
            }
            return rel;
        }

        public void UpdateRelation(string fromId, string toId, float affinityDelta, float trustDelta)
        {
            var rel = GetOrCreateRelation(fromId, toId);
            rel.affinity = Mathf.Clamp(rel.affinity + affinityDelta, -1f, 1f);
            rel.trust = Mathf.Clamp01(rel.trust + trustDelta);
            rel.interactionCount++;
            rel.lastInteraction = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            Save();
        }

        public void ShareMemory(string fromId, string toId, string memory)
        {
            var rel = GetOrCreateRelation(fromId, toId);
            rel.sharedMemories.Add(memory);
            if (rel.sharedMemories.Count > 20)
                rel.sharedMemories.RemoveAt(0);
            Save();
        }

        public string GetSocialSummary(string npcId, Dictionary<string, string> npcNames)
        {
            if (!_graph.ContainsKey(npcId)) return "No social connections.";

            var relations = _graph[npcId].relations
                .OrderByDescending(r => Mathf.Abs(r.affinity) + r.trust)
                .Take(5);

            var lines = new List<string>();
            foreach (var rel in relations)
            {
                string name = npcNames.ContainsKey(rel.targetId) ? npcNames[rel.targetId] : rel.targetId;
                string feeling = rel.affinity > 0.3f ? "friendly" :
                                 rel.affinity < -0.3f ? "hostile" : "neutral";
                string trustLevel = rel.trust > 0.7f ? "high trust" :
                                    rel.trust < 0.3f ? "low trust" : "moderate trust";
                lines.Add("- " + name + ": " + feeling + ", " + trustLevel + ", " + rel.interactionCount + " interactions");
            }

            return lines.Count > 0 ? string.Join("\n", lines) : "No significant relationships.";
        }

        public List<string> GetPropagationTargets(string fromId, float trustThreshold = 0.5f)
        {
            if (!_graph.ContainsKey(fromId)) return new List<string>();

            return _graph[fromId].relations
                .Where(r => r.trust >= trustThreshold && r.affinity > 0f)
                .OrderByDescending(r => r.trust)
                .Select(r => r.targetId)
                .ToList();
        }

        [Serializable]
        private class GraphSaveData
        {
            public List<SocialData> allData = new List<SocialData>();
        }

        public void Save()
        {
            if (string.IsNullOrEmpty(_savePath)) return;
            try
            {
                string dir = Path.GetDirectoryName(_savePath);
                if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
                var saveData = new GraphSaveData();
                saveData.allData.AddRange(_graph.Values);
                string json = JsonUtility.ToJson(saveData, true);
                File.WriteAllText(_savePath, json);
            }
            catch (Exception e)
            {
                Debug.LogWarning("[SocialGraph] Save failed: " + e.Message);
            }
        }

        private void Load()
        {
            if (string.IsNullOrEmpty(_savePath) || !File.Exists(_savePath)) return;
            try
            {
                string json = File.ReadAllText(_savePath);
                var saveData = JsonUtility.FromJson<GraphSaveData>(json);
                _graph.Clear();
                foreach (var data in saveData.allData)
                    _graph[data.npcId] = data;
            }
            catch (Exception e)
            {
                Debug.LogWarning("[SocialGraph] Load failed: " + e.Message);
            }
        }
    }
}
