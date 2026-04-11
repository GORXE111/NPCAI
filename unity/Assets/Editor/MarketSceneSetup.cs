#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

namespace NPCLLM.Editor
{
    public static class MarketSceneSetup
    {
        [MenuItem("NPCLLM/Setup 2D Market Scene")]
        public static void Setup()
        {
            // ── Camera: 2D top-down ────────────────
            var cam = Camera.main;
            if (cam != null)
            {
                cam.orthographic = true;
                cam.orthographicSize = 12f;
                cam.transform.position = new Vector3(0, 20, 0);
                cam.transform.rotation = Quaternion.Euler(90, 0, 0);
                cam.backgroundColor = new Color(0.56f, 0.49f, 0.36f); // dusty ground color
                EditorUtility.SetDirty(cam.gameObject);
            }

            // ── Ground ─────────────────────────────
            var ground = CreateQuad("Ground", Vector3.zero, new Vector3(30, 30, 1),
                Quaternion.Euler(90, 0, 0), new Color(0.65f, 0.55f, 0.40f));

            // ── Market Square (center area) ────────
            var square = CreateQuad("MarketSquare", new Vector3(0, 0.01f, 0), new Vector3(16, 12, 1),
                Quaternion.Euler(90, 0, 0), new Color(0.72f, 0.63f, 0.48f));

            // ── Stalls (10 stalls around the square) ──
            // Top row (5 stalls)
            CreateStall("Stall_Aldric",    new Vector3(-6, 0, 5),    "Blacksmith");
            CreateStall("Stall_Mira",      new Vector3(-3, 0, 5),    "Herbalist");
            CreateStall("Stall_Garrett",   new Vector3(0, 0, 5),     "General Store");
            CreateStall("Stall_Brynn",     new Vector3(3, 0, 5),     "Guard Post");
            CreateStall("Stall_Elara",     new Vector3(6, 0, 5),     "Rusty Tankard");

            // Bottom row (5 stalls)
            CreateStall("Stall_Thorne",       new Vector3(-6, 0, -5), "Hunter");
            CreateStall("Stall_SisterHelene", new Vector3(-3, 0, -5), "Chapel");
            CreateStall("Stall_Finn",         new Vector3(0, 0, -5),  "??? Corner");
            CreateStall("Stall_OldBertram",   new Vector3(3, 0, -5),  "Fish Market");
            CreateStall("Stall_Lydia",        new Vector3(6, 0, -5),  "Bakery");

            // ── NPC placeholders (colored circles) ───
            CreateNPCMarker("NPC_Aldric",       new Vector3(-6, 0.1f, 3.5f),  Color.red);
            CreateNPCMarker("NPC_Mira",         new Vector3(-3, 0.1f, 3.5f),  Color.green);
            CreateNPCMarker("NPC_Garrett",      new Vector3(0, 0.1f, 3.5f),   new Color(1f, 0.65f, 0f));
            CreateNPCMarker("NPC_Brynn",        new Vector3(3, 0.1f, 3.5f),   Color.blue);
            CreateNPCMarker("NPC_Elara",        new Vector3(6, 0.1f, 3.5f),   Color.magenta);
            CreateNPCMarker("NPC_Thorne",       new Vector3(-6, 0.1f, -3.5f), new Color(0.4f, 0.26f, 0.13f));
            CreateNPCMarker("NPC_SisterHelene", new Vector3(-3, 0.1f, -3.5f), Color.white);
            CreateNPCMarker("NPC_Finn",         new Vector3(0, 0.1f, -3.5f),  Color.yellow);
            CreateNPCMarker("NPC_OldBertram",   new Vector3(3, 0.1f, -3.5f),  Color.cyan);
            CreateNPCMarker("NPC_Lydia",        new Vector3(6, 0.1f, -3.5f),  new Color(1f, 0.8f, 0.6f));

            // ── Player marker ──────────────────────
            var playerGO = GameObject.Find("Player");
            if (playerGO == null)
                playerGO = new GameObject("Player");
            playerGO.transform.position = new Vector3(0, 0.1f, 0);
            var playerMarker = CreateDisc(playerGO, 0.6f, Color.white);
            // Add a small inner circle to distinguish player
            var inner = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            inner.name = "PlayerInner";
            inner.transform.SetParent(playerGO.transform);
            inner.transform.localPosition = new Vector3(0, 0.05f, 0);
            inner.transform.localScale = new Vector3(0.4f, 0.01f, 0.4f);
            SetColor(inner, Color.black);
            RemoveCollider(inner);

            // ── Decorations ────────────────────────
            // Well in center
            CreateDecor("Well", new Vector3(2, 0.1f, 0), 0.8f, new Color(0.5f, 0.5f, 0.55f));
            // Tree clusters
            CreateDecor("Tree1", new Vector3(-10, 0.1f, 8), 1.2f, new Color(0.2f, 0.5f, 0.2f));
            CreateDecor("Tree2", new Vector3(10, 0.1f, 8), 1.2f, new Color(0.2f, 0.5f, 0.2f));
            CreateDecor("Tree3", new Vector3(-10, 0.1f, -8), 1.0f, new Color(0.25f, 0.55f, 0.25f));
            CreateDecor("Tree4", new Vector3(10, 0.1f, -8), 1.0f, new Color(0.25f, 0.55f, 0.25f));

            Debug.Log("[MarketScene] 2D market scene created with 10 stalls, 10 NPC markers, and decorations");
        }

        private static GameObject CreateQuad(string name, Vector3 pos, Vector3 scale, Quaternion rot, Color color)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Quad);
            go.name = name;
            go.transform.position = pos;
            go.transform.localScale = scale;
            go.transform.rotation = rot;
            SetColor(go, color);
            RemoveCollider(go);
            return go;
        }

        private static void CreateStall(string name, Vector3 pos, string label)
        {
            var stall = new GameObject(name);
            stall.transform.position = pos;

            // Stall body (table)
            var table = GameObject.CreatePrimitive(PrimitiveType.Cube);
            table.name = "Table";
            table.transform.SetParent(stall.transform);
            table.transform.localPosition = Vector3.zero;
            table.transform.localScale = new Vector3(2.5f, 0.3f, 1.5f);
            SetColor(table, new Color(0.55f, 0.35f, 0.15f)); // wood brown
            RemoveCollider(table);

            // Roof/awning
            var roof = GameObject.CreatePrimitive(PrimitiveType.Cube);
            roof.name = "Roof";
            roof.transform.SetParent(stall.transform);
            roof.transform.localPosition = new Vector3(0, 1.5f, 0);
            roof.transform.localScale = new Vector3(2.8f, 0.1f, 1.8f);
            // Random warm color for roof
            float hue = Random.Range(0f, 0.12f); // red-orange range
            SetColor(roof, Color.HSVToRGB(hue, 0.6f, 0.8f));
            RemoveCollider(roof);

            // Support poles
            for (int i = -1; i <= 1; i += 2)
            {
                var pole = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                pole.name = "Pole";
                pole.transform.SetParent(stall.transform);
                pole.transform.localPosition = new Vector3(i * 1.1f, 0.75f, 0);
                pole.transform.localScale = new Vector3(0.1f, 0.75f, 0.1f);
                SetColor(pole, new Color(0.45f, 0.3f, 0.12f));
                RemoveCollider(pole);
            }
        }

        private static void CreateNPCMarker(string name, Vector3 pos, Color color)
        {
            var go = new GameObject(name);
            go.transform.position = pos;
            CreateDisc(go, 0.5f, color);
        }

        private static GameObject CreateDisc(GameObject parent, float radius, Color color)
        {
            var disc = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            disc.name = "Marker";
            disc.transform.SetParent(parent.transform);
            disc.transform.localPosition = Vector3.zero;
            disc.transform.localScale = new Vector3(radius, 0.02f, radius);
            SetColor(disc, color);
            RemoveCollider(disc);
            return disc;
        }

        private static void CreateDecor(string name, Vector3 pos, float size, Color color)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            go.name = name;
            go.transform.position = pos;
            go.transform.localScale = new Vector3(size, size * 0.5f, size);
            SetColor(go, color);
            RemoveCollider(go);
        }

        private static void SetColor(GameObject go, Color color)
        {
            var renderer = go.GetComponent<Renderer>();
            if (renderer != null)
            {
                // Use URP Lit shader
                var mat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
                mat.color = color;
                renderer.sharedMaterial = mat;
            }
        }

        private static void RemoveCollider(GameObject go)
        {
            var col = go.GetComponent<Collider>();
            if (col != null) Object.DestroyImmediate(col);
        }
    }
}
#endif
