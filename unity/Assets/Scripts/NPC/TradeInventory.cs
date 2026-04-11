using System;
using System.Collections.Generic;
using UnityEngine;

namespace NPCLLM.NPC
{
    [Serializable]
    public class TradeItem
    {
        public string itemName;
        public int quantity;
        public int basePrice;
    }

    /// <summary>
    /// Simple trading inventory for NPC shops.
    /// </summary>
    public class TradeInventory : MonoBehaviour
    {
        [SerializeField] private List<TradeItem> items = new List<TradeItem>();

        public List<TradeItem> GetAvailableItems()
        {
            return items.FindAll(i => i.quantity > 0);
        }

        public bool TryBuy(string itemName, out int price)
        {
            price = 0;
            var item = items.Find(i => i.itemName == itemName && i.quantity > 0);
            if (item == null) return false;

            price = item.basePrice;
            item.quantity--;
            return true;
        }

        public void SetItems(List<TradeItem> newItems)
        {
            items = newItems;
        }

        public string GetInventoryString()
        {
            var available = GetAvailableItems();
            if (available.Count == 0) return "Nothing for sale.";

            var lines = new List<string>();
            foreach (var item in available)
                lines.Add(item.itemName + " - " + item.basePrice + " gold (x" + item.quantity + ")");
            return string.Join("\n", lines);
        }
    }
}
