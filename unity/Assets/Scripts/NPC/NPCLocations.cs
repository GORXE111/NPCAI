using System.Collections.Generic;
using UnityEngine;

namespace NPCLLM.NPC
{
    /// <summary>
    /// Static location data for the market town.
    /// Matches positions from MarketSceneSetup.cs.
    /// </summary>
    public static class NPCLocations
    {
        // Stall positions (where NPC stands in front of their stall)
        public static readonly Dictionary<string, Vector3> StallPositions = new Dictionary<string, Vector3>
        {
            { "Aldric",        new Vector3(-6f, 0.1f, 3.5f) },
            { "Mira",          new Vector3(-3f, 0.1f, 3.5f) },
            { "Garrett",       new Vector3(0f, 0.1f, 3.5f) },
            { "Brynn",         new Vector3(3f, 0.1f, 3.5f) },
            { "Elara",         new Vector3(6f, 0.1f, 3.5f) },
            { "Thorne",        new Vector3(-6f, 0.1f, -3.5f) },
            { "Sister Helene", new Vector3(-3f, 0.1f, -3.5f) },
            { "Finn",          new Vector3(0f, 0.1f, -3.5f) },
            { "Old Bertram",   new Vector3(3f, 0.1f, -3.5f) },
            { "Lydia",         new Vector3(6f, 0.1f, -3.5f) }
        };

        // Town square center
        public static readonly Vector3 SquareCenter = new Vector3(0f, 0.1f, 0f);

        // Well position
        public static readonly Vector3 WellPosition = new Vector3(2f, 0.1f, 0f);

        // Wander points (places NPCs might walk to during idle time)
        public static readonly Vector3[] WanderPoints = new Vector3[]
        {
            new Vector3(-4f, 0.1f, 0f),
            new Vector3(4f, 0.1f, 0f),
            new Vector3(0f, 0.1f, 2f),
            new Vector3(0f, 0.1f, -2f),
            new Vector3(-2f, 0.1f, 1f),
            new Vector3(2f, 0.1f, -1f),
            new Vector3(-5f, 0.1f, 1.5f),
            new Vector3(5f, 0.1f, -1.5f),
        };

        public static Vector3 GetRandomWanderPoint()
        {
            return WanderPoints[Random.Range(0, WanderPoints.Length)];
        }

        public static Vector3 GetStallPosition(string npcName)
        {
            return StallPositions.ContainsKey(npcName)
                ? StallPositions[npcName]
                : SquareCenter;
        }
    }
}
