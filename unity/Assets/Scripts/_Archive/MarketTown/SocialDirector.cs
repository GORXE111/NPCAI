using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace NPCLLM.NPC
{
    /// <summary>
    /// Manages autonomous NPC-NPC conversations.
    /// World rule: NPC must walk close to another NPC before chatting.
    /// Flow: Pick pair -> Initiator walks to responder -> Arrive -> Chat -> Both return
    /// </summary>
    public class SocialDirector : MonoBehaviour
    {
        public static SocialDirector Instance { get; private set; }

        [Header("Settings")]
        [SerializeField] private float globalCooldown = 30f;
        [SerializeField] private float pairCooldown = 120f;
        [SerializeField] private float tickInterval = 10f;
        [SerializeField] private int maxQueueDepth = 2;
        [SerializeField] private float chatDistance = 1.5f; // must be this close to chat

        [Header("Stats")]
        [SerializeField] private int totalConversations;
        [SerializeField] private string lastConversation;

        private float _lastChatTime = -999f;
        private float _tickTimer;
        private readonly Dictionary<string, float> _pairLastChat = new Dictionary<string, float>();
        private List<NPCBrain> _allNPCs;

        // Pending approach: initiator is walking toward responder
        private PendingChat _pendingChat;

        private class PendingChat
        {
            public NPCBrain initiator;
            public NPCBrain responder;
            public NPCBehavior initBehavior;
            public NPCBehavior respBehavior;
            public string openingLine;
            public bool initiated;
        }

        private static readonly string[] GOSSIP_TEMPLATES = new string[]
        {
            "Have you noticed anything strange lately?",
            "Business has been {quality} today, wouldn't you say?",
            "I heard something interesting the other day...",
            "What do you think about the new taxes the king is raising?",
            "The weather has been odd lately, don't you think?",
            "Did you see that stranger who came through yesterday?",
            "I've been thinking about the wolves near the border...",
            "Someone mentioned the storehouse has been losing supplies.",
            "Have you visited the old ruins recently?",
            "The river's been running strange colors, I hear."
        };

        private static readonly string[] QUALITY_WORDS = { "slow", "busy", "quiet", "lively" };

        private void Awake()
        {
            if (Instance != null && Instance != this) { Destroy(this); return; }
            Instance = this;
        }

        public void Initialize(List<NPCBrain> npcs)
        {
            _allNPCs = npcs;
        }

        private void Update()
        {
            if (_allNPCs == null || _allNPCs.Count < 2) return;

            // Check pending approach
            if (_pendingChat != null)
            {
                UpdatePendingChat();
                return; // Don't start new conversations while one is pending
            }

            _tickTimer += Time.deltaTime;
            if (_tickTimer < tickInterval) return;
            _tickTimer = 0f;

            TryInitiateConversation();
        }

        private void UpdatePendingChat()
        {
            var pc = _pendingChat;

            // Check if initiator arrived close enough
            float dist = Vector3.Distance(
                pc.initiator.transform.position,
                pc.responder.transform.position);

            if (dist <= chatDistance)
            {
                // Arrived! Start the actual chat
                if (!pc.initiated)
                {
                    pc.initiated = true;
                    StartActualChat(pc);
                }
            }
            else if (pc.initBehavior.CurrentState != NPCState.Walking)
            {
                // Initiator stopped walking but isn't close enough - re-walk
                pc.initBehavior.WalkTo(pc.responder.transform.position, NPCState.Idle);
            }
        }

        private void TryInitiateConversation()
        {
            if (Time.realtimeSinceStartup - _lastChatTime < globalCooldown) return;

            if (NPCScheduler.Instance != null && NPCScheduler.Instance.TotalRequests - totalConversations > maxQueueDepth)
                return;

            var pair = FindBestPair();
            if (pair == null) return;

            var (initiator, responder) = pair.Value;
            var initBehavior = initiator.GetComponent<NPCBehavior>();
            var respBehavior = responder.GetComponent<NPCBehavior>();
            if (initBehavior == null || respBehavior == null) return;

            string openingLine = GenerateOpeningLine(initiator);

            // Check if already close enough
            float dist = Vector3.Distance(initiator.transform.position, responder.transform.position);
            if (dist <= chatDistance)
            {
                // Already close, chat immediately
                _pendingChat = new PendingChat
                {
                    initiator = initiator, responder = responder,
                    initBehavior = initBehavior, respBehavior = respBehavior,
                    openingLine = openingLine, initiated = false
                };
                _pendingChat.initiated = true;
                StartActualChat(_pendingChat);
            }
            else
            {
                // Walk toward responder first
                Debug.Log("[SocialDirector] " + initiator.NpcName + " walking toward " + responder.NpcName);
                initBehavior.WalkTo(responder.transform.position, NPCState.Idle);

                _pendingChat = new PendingChat
                {
                    initiator = initiator, responder = responder,
                    initBehavior = initBehavior, respBehavior = respBehavior,
                    openingLine = openingLine, initiated = false
                };
            }
        }

        private void StartActualChat(PendingChat pc)
        {
            // Lock both in chatting state
            pc.initBehavior.StartChat(pc.respBehavior);
            pc.respBehavior.StartChat(pc.initBehavior);

            _lastChatTime = Time.realtimeSinceStartup;
            string pairKey = GetPairKey(pc.initiator.NpcId, pc.responder.NpcId);
            _pairLastChat[pairKey] = Time.realtimeSinceStartup;
            totalConversations++;

            Debug.Log("[SocialDirector] " + pc.initiator.NpcName + " -> " + pc.responder.NpcName + ": " + pc.openingLine);

            // Show initiator's opening line in bubble (no LLM)
            pc.initiator.ShowSpeech(pc.openingLine);

            // Only responder uses LLM
            NPCScheduler.Instance.RequestNPCChat(pc.initiator, pc.responder, pc.openingLine, response =>
            {
                lastConversation = pc.initiator.NpcName + " -> " + pc.responder.NpcName;
                Debug.Log("[SocialDirector] " + pc.responder.NpcName + " replies: " +
                    (response.Length > 80 ? response.Substring(0, 80) + "..." : response));

                // End chatting state, initiator walks back home
                pc.initBehavior.EndChat();
                pc.respBehavior.EndChat();
                pc.initBehavior.WalkTo(pc.initBehavior.HomePosition, NPCState.Working);

                _pendingChat = null;
            });
        }

        private (NPCBrain, NPCBrain)? FindBestPair()
        {
            var available = _allNPCs
                .Where(n => {
                    var b = n.GetComponent<NPCBehavior>();
                    return b != null && b.IsAvailableForChat() && !n.IsBusy;
                })
                .ToList();

            if (available.Count < 2) return null;

            float bestScore = -1f;
            NPCBrain bestA = null, bestB = null;

            for (int i = 0; i < available.Count; i++)
            {
                for (int j = i + 1; j < available.Count; j++)
                {
                    var a = available[i];
                    var b = available[j];

                    string pairKey = GetPairKey(a.NpcId, b.NpcId);
                    if (_pairLastChat.ContainsKey(pairKey) &&
                        Time.realtimeSinceStartup - _pairLastChat[pairKey] < pairCooldown)
                        continue;

                    var relation = SocialGraph.Instance.GetRelation(a.NpcId, b.NpcId);
                    float affinity = relation != null ? Mathf.Abs(relation.affinity) + 0.1f : 0.1f;
                    float dist = Vector3.Distance(a.transform.position, b.transform.position);
                    float proximity = 1f / (1f + dist * 0.5f);

                    float score = affinity * proximity;
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestA = a;
                        bestB = b;
                    }
                }
            }

            if (bestA == null) return null;
            return (bestA, bestB);
        }

        private string GenerateOpeningLine(NPCBrain initiator)
        {
            if (initiator.Personality != null &&
                initiator.Personality.knowledge != null &&
                initiator.Personality.knowledge.Count > 0 &&
                Random.value < 0.4f)
            {
                var knowledge = initiator.Personality.knowledge;
                string topic = knowledge[Random.Range(0, knowledge.Count)];
                return "You know, " + topic.ToLower() + ". What do you think about that?";
            }

            string line = GOSSIP_TEMPLATES[Random.Range(0, GOSSIP_TEMPLATES.Length)];
            line = line.Replace("{quality}", QUALITY_WORDS[Random.Range(0, QUALITY_WORDS.Length)]);
            return line;
        }

        private static string GetPairKey(string id1, string id2)
        {
            return string.Compare(id1, id2) < 0 ? id1 + ":" + id2 : id2 + ":" + id1;
        }

        public int TotalConversations => totalConversations;
    }
}
