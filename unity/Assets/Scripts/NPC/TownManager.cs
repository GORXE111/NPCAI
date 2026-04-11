using System.Collections.Generic;
using UnityEngine;
using NPCLLM.LLM;
using NPCLLM.NPC;

namespace NPCLLM
{
    public class TownManager : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private Transform player;
        [SerializeField] private OllamaClient ollamaClient;

        [Header("Runtime")]
        [SerializeField] private List<NPCBrain> allNPCs = new List<NPCBrain>();

        private static readonly NPCPersonality[] NPC_TEMPLATES = new NPCPersonality[]
        {
            new NPCPersonality { npcName = "Aldric", occupation = "blacksmith",
                personality = "Gruff but kind-hearted. Takes pride in his craft. Protective of the town.",
                speechStyle = "Short, direct sentences. Uses forge metaphors.",
                backstory = "Has been the town's blacksmith for 20 years. Lost his wife to illness 3 years ago.",
                knowledge = new List<string> { "Sells swords, shields, and armor", "Knows every metal and alloy", "Heard rumors about bandits in the north" } },
            new NPCPersonality { npcName = "Mira", occupation = "herbalist",
                personality = "Warm and nurturing. Speaks softly. Believes in natural remedies.",
                speechStyle = "Gentle, uses plant metaphors. Sometimes cryptic.",
                backstory = "Traveled from a distant land 5 years ago. Keeps to herself but helps anyone who is sick.",
                knowledge = new List<string> { "Sells potions, herbs, and remedies", "Knows healing recipes", "Senses something dark in the forest lately" } },
            new NPCPersonality { npcName = "Garrett", occupation = "merchant",
                personality = "Shrewd and calculating. Always looking for a deal. Secretly generous.",
                speechStyle = "Smooth-talking, uses numbers and trade terms.",
                backstory = "Runs the general store. Has trade connections to three neighboring towns.",
                knowledge = new List<string> { "Sells food, tools, and supplies", "Knows market prices", "Recently heard the king is raising taxes" } },
            new NPCPersonality { npcName = "Brynn", occupation = "guard captain",
                personality = "Disciplined and loyal. Follows rules strictly. Worries about the town's safety.",
                speechStyle = "Formal, military tone. Brief and authoritative.",
                backstory = "Served in the royal army before being posted here. Respects honor above all.",
                knowledge = new List<string> { "Patrols the town perimeter", "Knows about recent wolf sightings", "Suspects someone is stealing from the storehouse" } },
            new NPCPersonality { npcName = "Elara", occupation = "innkeeper",
                personality = "Cheerful gossip. Knows everyone's business. Loves stories.",
                speechStyle = "Chatty, warm, uses endearments like 'dear' and 'love'.",
                backstory = "Inherited the Rusty Tankard inn from her father. The social hub of town.",
                knowledge = new List<string> { "Sells food, drink, and lodging", "Hears all the town gossip", "A stranger arrived last night asking odd questions" } },
            new NPCPersonality { npcName = "Thorne", occupation = "hunter",
                personality = "Quiet and observant. Prefers the wilderness to people. Loyal to few.",
                speechStyle = "Few words, precise. Speaks of nature and tracking.",
                backstory = "Lives at the edge of town near the forest. Supplies the town with game and pelts.",
                knowledge = new List<string> { "Sells pelts, meat, and bows", "Tracks animals and people", "Found strange footprints near the old ruins" } },
            new NPCPersonality { npcName = "Sister Helene", occupation = "priestess",
                personality = "Compassionate and wise. Speaks with conviction. Mediates disputes.",
                speechStyle = "Calm, measured, occasionally quotes scripture.",
                backstory = "Tends the small chapel at the edge of the square. People trust her judgment.",
                knowledge = new List<string> { "Offers blessings and counsel", "Knows the town's history", "Has been having troubling visions" } },
            new NPCPersonality { npcName = "Finn", occupation = "pickpocket",
                personality = "Cheeky and quick-witted. Streetwise. Has a good heart under the bravado.",
                speechStyle = "Slang, fast-paced, deflects with humor.",
                backstory = "Orphan who grew up on the streets. Steals to survive but never from the poor.",
                knowledge = new List<string> { "Knows secret passages in town", "Overheard a shady deal at the docks", "Can get 'special items' for a price" } },
            new NPCPersonality { npcName = "Old Bertram", occupation = "fisherman",
                personality = "Patient and philosophical. Tells long stories. Wise in his own way.",
                speechStyle = "Slow, reflective, uses sea metaphors and old sayings.",
                backstory = "Has fished the river for 50 years. Claims to have seen a sea serpent once.",
                knowledge = new List<string> { "Sells fresh fish", "Knows weather patterns", "The river has been running strange colors upstream" } },
            new NPCPersonality { npcName = "Lydia", occupation = "baker",
                personality = "Warm and motherly. Practical and hardworking. Fiercely protective of children.",
                speechStyle = "Homey, uses food metaphors, straightforward.",
                backstory = "Wakes before dawn every day. Her bread is famous three towns over.",
                knowledge = new List<string> { "Sells bread, pastries, and pies", "Knows everyone's favorite food", "Noticed the flour shipments have been smaller lately" } }
        };

        // Trade inventories per occupation
        private static readonly Dictionary<string, List<TradeItem>> TRADE_ITEMS = new Dictionary<string, List<TradeItem>>
        {
            { "Aldric", new List<TradeItem> {
                new TradeItem { itemName = "Iron Sword", quantity = 3, basePrice = 50 },
                new TradeItem { itemName = "Steel Shield", quantity = 2, basePrice = 75 },
                new TradeItem { itemName = "Chain Armor", quantity = 1, basePrice = 120 } } },
            { "Mira", new List<TradeItem> {
                new TradeItem { itemName = "Healing Potion", quantity = 5, basePrice = 15 },
                new TradeItem { itemName = "Antidote", quantity = 3, basePrice = 10 },
                new TradeItem { itemName = "Dried Herbs", quantity = 10, basePrice = 5 } } },
            { "Garrett", new List<TradeItem> {
                new TradeItem { itemName = "Rope", quantity = 5, basePrice = 8 },
                new TradeItem { itemName = "Lantern", quantity = 3, basePrice = 12 },
                new TradeItem { itemName = "Map", quantity = 2, basePrice = 20 } } },
            { "Elara", new List<TradeItem> {
                new TradeItem { itemName = "Ale", quantity = 10, basePrice = 3 },
                new TradeItem { itemName = "Stew", quantity = 5, basePrice = 5 },
                new TradeItem { itemName = "Room for the Night", quantity = 3, basePrice = 15 } } },
            { "Thorne", new List<TradeItem> {
                new TradeItem { itemName = "Deer Pelt", quantity = 4, basePrice = 18 },
                new TradeItem { itemName = "Shortbow", quantity = 2, basePrice = 40 },
                new TradeItem { itemName = "Venison", quantity = 6, basePrice = 8 } } },
            { "Finn", new List<TradeItem> {
                new TradeItem { itemName = "Lockpick Set", quantity = 3, basePrice = 25 },
                new TradeItem { itemName = "Smoke Bomb", quantity = 2, basePrice = 15 } } },
            { "Old Bertram", new List<TradeItem> {
                new TradeItem { itemName = "Fresh Fish", quantity = 8, basePrice = 4 },
                new TradeItem { itemName = "Fishing Rod", quantity = 2, basePrice = 20 } } },
            { "Lydia", new List<TradeItem> {
                new TradeItem { itemName = "Fresh Bread", quantity = 10, basePrice = 2 },
                new TradeItem { itemName = "Meat Pie", quantity = 5, basePrice = 6 },
                new TradeItem { itemName = "Sweet Roll", quantity = 8, basePrice = 3 } } }
        };

        private void Start()
        {
            if (ollamaClient == null)
                ollamaClient = FindFirstObjectByType<OllamaClient>();

            // Ensure DayClock exists
            if (DayClock.Instance == null)
                gameObject.AddComponent<DayClock>();

            // Ensure SocialDirector exists
            if (SocialDirector.Instance == null)
                gameObject.AddComponent<SocialDirector>();

            SetupTown();

            // Initialize systems
            if (NPCScheduler.Instance != null && player != null)
                NPCScheduler.Instance.SetPlayer(player);

            SocialDirector.Instance.Initialize(allNPCs);

            Debug.Log("[TownManager] Town initialized with " + allNPCs.Count + " NPCs");
        }

        private void SetupTown()
        {
            for (int i = 0; i < NPC_TEMPLATES.Length; i++)
            {
                var template = NPC_TEMPLATES[i];

                // Use stall positions from NPCLocations
                Vector3 pos = NPCLocations.GetStallPosition(template.npcName);

                GameObject npcGO = new GameObject("NPC_" + template.npcName);
                npcGO.transform.position = pos;

                // Add visual marker (colored cylinder)
                var marker = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                marker.name = "Marker";
                marker.transform.SetParent(npcGO.transform);
                marker.transform.localPosition = Vector3.zero;
                marker.transform.localScale = new Vector3(0.5f, 0.02f, 0.5f);
                var col = marker.GetComponent<Collider>();
                if (col != null) Destroy(col);
                var renderer = marker.GetComponent<Renderer>();
                if (renderer != null)
                {
                    var mat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
                    mat.color = GetNPCColor(i);
                    renderer.sharedMaterial = mat;
                }

                // Core components
                var brain = npcGO.AddComponent<NPCBrain>();
                brain.SetPersonality(template);
                brain.SetNameColor(GetNPCColor(i));

                var behavior = npcGO.AddComponent<NPCBehavior>();
                behavior.Initialize(template.npcName, template.occupation, pos);

                // Trade inventory
                if (TRADE_ITEMS.ContainsKey(template.npcName))
                {
                    var inventory = npcGO.AddComponent<TradeInventory>();
                    inventory.SetItems(TRADE_ITEMS[template.npcName]);
                }

                allNPCs.Add(brain);

                Debug.Log("[TownManager] Spawned " + template.npcName + " (" + template.occupation + ") at " + pos);
            }

            // Setup initial social relationships
            SetRelation("Aldric", "Lydia", 0.5f, 0.6f);
            SetRelation("Elara", "Finn", 0.2f, 0.3f);
            SetRelation("Brynn", "Finn", -0.4f, 0.1f);
            SetRelation("Mira", "Sister Helene", 0.4f, 0.5f);
            SetRelation("Thorne", "Old Bertram", 0.3f, 0.4f);
            SetRelation("Garrett", "Elara", 0.3f, 0.5f);

            // Neighbors
            for (int i = 0; i < allNPCs.Count; i++)
            {
                int prev = (i - 1 + allNPCs.Count) % allNPCs.Count;
                int next = (i + 1) % allNPCs.Count;
                SocialGraph.Instance.UpdateRelation(allNPCs[i].NpcId, allNPCs[prev].NpcId, 0.2f, 0.15f);
                SocialGraph.Instance.UpdateRelation(allNPCs[i].NpcId, allNPCs[next].NpcId, 0.2f, 0.15f);
            }
        }

        private void SetRelation(string name1, string name2, float affinity, float trust)
        {
            var npc1 = allNPCs.Find(n => n.NpcName == name1);
            var npc2 = allNPCs.Find(n => n.NpcName == name2);
            if (npc1 != null && npc2 != null)
            {
                SocialGraph.Instance.UpdateRelation(npc1.NpcId, npc2.NpcId, affinity, trust);
                SocialGraph.Instance.UpdateRelation(npc2.NpcId, npc1.NpcId, affinity, trust);
            }
        }

        private static Color GetNPCColor(int index)
        {
            Color[] colors = { Color.red, Color.green, new Color(1f, 0.65f, 0f), Color.blue, Color.magenta,
                new Color(0.4f, 0.26f, 0.13f), Color.white, Color.yellow, Color.cyan, new Color(1f, 0.8f, 0.6f) };
            return colors[index % colors.Length];
        }

        // ── Public API ─────────────────────────────

        public void StartPropagationExperiment(string sourceNpcName, string information)
        {
            var sourceNpc = allNPCs.Find(n => n.NpcName == sourceNpcName);
            if (sourceNpc == null) { Debug.LogError("[Experiment] NPC not found: " + sourceNpcName); return; }

            Debug.Log("[Experiment] Starting propagation from " + sourceNpcName + ": " + information);
            sourceNpc.Memory.AddMemory("Important news: " + information, "episodic", 0.9f);

            var targets = SocialGraph.Instance.GetPropagationTargets(sourceNpc.NpcId, 0.4f);
            foreach (var targetId in targets)
            {
                var targetNpc = allNPCs.Find(n => n.NpcId == targetId);
                if (targetNpc != null)
                {
                    NPCScheduler.Instance.RequestNPCChat(sourceNpc, targetNpc, "I have news: " + information,
                        response => Debug.Log("[Propagation] " + sourceNpcName + " -> " + targetNpc.NpcName + ": " + response));
                }
            }
        }

        public List<NPCBrain> GetAllNPCs() => allNPCs;

        public void TalkToNPC(string npcName, string message, System.Action<string> onResponse)
        {
            var npc = allNPCs.Find(n => n.NpcName == npcName);
            if (npc == null) { onResponse?.Invoke("NPC not found: " + npcName); return; }

            if (NPCScheduler.Instance != null)
                NPCScheduler.Instance.RequestPlayerChat(npc, message, onResponse);
            else
                npc.TalkTo(message, onResponse);
        }
    }
}
