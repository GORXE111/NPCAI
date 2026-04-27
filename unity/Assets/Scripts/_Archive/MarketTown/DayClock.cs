using System;
using UnityEngine;

namespace NPCLLM.NPC
{
    public enum TimePeriod { Dawn, Morning, Afternoon, Evening, Night }

    /// <summary>
    /// Game time system. 1 real minute = 1 game hour (full day = 24 real minutes).
    /// </summary>
    public class DayClock : MonoBehaviour
    {
        public static DayClock Instance { get; private set; }

        [Header("Settings")]
        [SerializeField] private float realSecondsPerGameHour = 60f;
        [SerializeField] private float startHour = 7f;

        [Header("State")]
        [SerializeField] private float currentHour;
        [SerializeField] private TimePeriod currentPeriod;
        [SerializeField] private int dayCount;

        public float CurrentHour => currentHour;
        public TimePeriod CurrentPeriod => currentPeriod;
        public int DayCount => dayCount;

        public event Action<TimePeriod> OnPeriodChanged;

        private void Awake()
        {
            if (Instance != null && Instance != this) { Destroy(this); return; }
            Instance = this;
            currentHour = startHour;
            currentPeriod = GetPeriod(currentHour);
            dayCount = 1;
        }

        private void Update()
        {
            float prevHour = currentHour;
            currentHour += Time.deltaTime / realSecondsPerGameHour;

            if (currentHour >= 24f)
            {
                currentHour -= 24f;
                dayCount++;
                Debug.Log("[DayClock] Day " + dayCount + " begins");
            }

            TimePeriod newPeriod = GetPeriod(currentHour);
            if (newPeriod != currentPeriod)
            {
                currentPeriod = newPeriod;
                Debug.Log("[DayClock] " + currentHour.ToString("F1") + "h - Period: " + currentPeriod);
                OnPeriodChanged?.Invoke(currentPeriod);
            }
        }

        public static TimePeriod GetPeriod(float hour)
        {
            if (hour >= 5f && hour < 7f) return TimePeriod.Dawn;
            if (hour >= 7f && hour < 12f) return TimePeriod.Morning;
            if (hour >= 12f && hour < 17f) return TimePeriod.Afternoon;
            if (hour >= 17f && hour < 20f) return TimePeriod.Evening;
            return TimePeriod.Night;
        }

        /// <summary>
        /// Get display string like "Day 1 - 14:30 (Afternoon)"
        /// </summary>
        public string GetTimeString()
        {
            int h = Mathf.FloorToInt(currentHour);
            int m = Mathf.FloorToInt((currentHour - h) * 60f);
            return "Day " + dayCount + " - " + h.ToString("D2") + ":" + m.ToString("D2") + " (" + currentPeriod + ")";
        }
    }
}
