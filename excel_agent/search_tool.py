from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


_OFFLINE_DICTIONARY: Dict[str, Dict[str, Any]] = {
    "inventory turnover": {
        "definition": "Inventory turnover is a financial ratio measuring how many times inventory is sold and replaced during a period. Formula: Cost of Goods Sold / Average Inventory. Higher turnover indicates efficient inventory management.",
        "formula": "Inventory Turnover Ratio = Cost of Goods Sold (COGS) / Average Inventory Value",
        "category": "Financial Ratio",
    },
    "stock keeping unit": {
        "definition": "A Stock Keeping Unit (SKU) is a unique scannable code assigned to each distinct product variant in a retailer's inventory, used to automatically track inventory movement and levels.",
        "category": "Inventory Management",
    },
    "sku": {
        "definition": "SKU (Stock Keeping Unit) is a unique identifier for each product variant, enabling automatic inventory tracking and management.",
        "category": "Inventory Management",
    },
    "safety stock": {
        "definition": "Safety stock is the buffer quantity of inventory held to protect against demand volatility, supply delays, or forecasting errors. It prevents stockouts between replenishment cycles.",
        "formula": "Safety Stock ≈ Z × σLT × D_avg, where Z = service level factor, σLT = lead time std dev, D_avg = avg demand",
        "category": "Inventory Management",
    },
    "opening stock": {
        "definition": "Opening stock (or beginning inventory) is the value or quantity of goods held by a business at the start of an accounting period. It is carried forward from the closing stock of the prior period.",
        "category": "Accounting / Inventory",
    },
    "closing stock": {
        "definition": "Closing stock (or ending inventory / hand-in-stock) is the value or quantity of unsold goods remaining at the end of an accounting period. Formula: Opening Stock + Purchases - Units Sold.",
        "formula": "Closing Stock = Opening Stock + Stock In - Units Sold",
        "category": "Accounting / Inventory",
    },
    "hand-in-stock": {
        "definition": "Hand-in-Stock (on-hand inventory or closing stock) is the quantity of product currently physically available in the warehouse. Calculated as Opening Stock + Purchases - Units Sold.",
        "formula": "Hand-In-Stock = Opening Stock + Purchase/Stock In - Number of Units Sold",
        "category": "Inventory Management",
    },
    "cost of goods sold": {
        "definition": "Cost of Goods Sold (COGS) is the direct cost of manufacturing or acquiring the products sold during a period. It includes material, labor, and allocated overhead; it excludes indirect expenses like marketing.",
        "formula": "COGS = Opening Inventory + Purchases - Closing Inventory",
        "category": "Accounting",
    },
    "cogs": {
        "definition": "COGS (Cost of Goods Sold) is the direct cost of producing/acquiring goods sold in a period: Opening Inventory + Purchases - Closing Inventory.",
        "category": "Accounting",
    },
    "gross profit": {
        "definition": "Gross profit is revenue from sales minus the Cost of Goods Sold. It represents profit from core trading before deducting operating expenses like overhead, salaries, and taxes.",
        "formula": "Gross Profit = Revenue - Cost of Goods Sold",
        "category": "Finance",
    },
    "gross margin": {
        "definition": "Gross margin (gross profit margin) is gross profit expressed as a percentage of revenue. It measures how efficiently a company converts sales into profit on product cost.",
        "formula": "Gross Margin (%) = (Gross Profit / Revenue) × 100",
        "category": "Financial Ratio",
    },
    "reorder point": {
        "definition": "The reorder point is the inventory level at which a new purchase order should be placed to replenish stock before it runs out. It accounts for lead time demand and safety stock.",
        "formula": "Reorder Point = (Average Daily Demand × Lead Time in Days) + Safety Stock",
        "category": "Inventory Management",
    },
    "eoq": {
        "definition": "Economic Order Quantity (EOQ) is the optimal order size that minimizes total inventory costs (ordering + holding + shortage costs).",
        "formula": "EOQ = √(2 × D × S / H), where D = annual demand, S = order cost per order, H = holding cost per unit/year",
        "category": "Inventory Management",
    },
    "economic order quantity": {
        "definition": "Economic Order Quantity (EOQ) is the order quantity that minimizes the sum of ordering, holding, and shortage costs over a period.",
        "formula": "EOQ = √(2DS / H)",
        "category": "Inventory Management",
    },
    "abc analysis": {
        "definition": "ABC analysis classifies inventory into three categories: A (high-value, low-quantity ~20% SKUs = 80% value), B (moderate), C (low-value, high-quantity). Used to prioritize management effort.",
        "category": "Inventory Classification",
    },
    "lead time": {
        "definition": "Lead time is the total time elapsed between placing a purchase order and receiving the goods into stock. It includes order processing, manufacturing/shipping, and receiving time.",
        "category": "Supply Chain",
    },
    "stockout": {
        "definition": "A stockout occurs when customer demand cannot be fulfilled from available on-hand inventory, resulting in lost sales, backorders, or customer dissatisfaction.",
        "category": "Inventory Risk",
    },
    "obsolescence": {
        "definition": "Inventory obsolescence is when stock becomes outdated, expired, or unsellable due to changing demand, new product versions, or expiry dates. Usually requires a write-down.",
        "category": "Inventory Risk",
    },
    "carrying cost": {
        "definition": "Carrying cost (holding cost) is the total cost of holding inventory over time, including warehousing, insurance, depreciation, obsolescence, and capital opportunity cost. Often 15-35% of inventory value per year.",
        "category": "Inventory Cost",
    },
    "revenue": {
        "definition": "Revenue (sales revenue / top line) is the total income generated from selling goods or services before expenses are deducted. Formula: Units Sold × Selling Price per Unit.",
        "formula": "Revenue = Number of Units Sold × Unit Selling Price",
        "category": "Finance",
    },
    "average inventory": {
        "definition": "Average inventory is the mean value of inventory over a period, typically computed as (Opening Stock + Closing Stock) / 2. Used in ratios like inventory turnover.",
        "formula": "Average Inventory = (Opening Stock + Closing Stock) / 2",
        "category": "Inventory / Finance",
    },
    "days inventory outstanding": {
        "definition": "Days Inventory Outstanding (DIO / days sales of inventory) measures the average number of days inventory is held before sale. Lower DIO indicates faster-moving stock.",
        "formula": "DIO = (Average Inventory / COGS) × 365",
        "category": "Financial Ratio",
    },
    "dio": {
        "definition": "DIO (Days Inventory Outstanding) = (Average Inventory / COGS) × 365. Average days inventory sits on shelves before sale.",
        "category": "Financial Ratio",
    },
    "value of inventory": {
        "definition": "Total inventory value is the sum across all SKUs of (Units on-Hand × Cost Price per Unit). Represented on the balance sheet as a current asset.",
        "formula": "Total Inventory Value = Σ (Hand-In-Stock Quantity × Cost Price Per Unit)",
        "category": "Accounting / Inventory",
    },
}


@dataclass
class SearchResult:
    query: str
    success: bool
    definition: str = ""
    category: str = ""
    formula: Optional[str] = None
    sources_used: List[str] = field(default_factory=list)
    note: str = ""

    def __bool__(self) -> bool:
        return bool(self.definition)


class SearchTool:
    def __init__(self, web_search_fn: Optional[Callable[[str], List[str]]] = None):
        self._dict = _OFFLINE_DICTIONARY
        self._web_search_fn = web_search_fn
        self.history: List[SearchResult] = []

    def _normalize(self, term: str) -> str:
        t = re.sub(r"[^a-z0-9\s/]", "", term.lower()).strip()
        t = re.sub(r"\s+", " ", t)
        return t

    def lookup(self, query: str) -> SearchResult:
        q_norm = self._normalize(query)
        if not q_norm:
            res = SearchResult(query=query, success=False, note="Empty query.")
            self.history.append(res)
            return res

        best_key = None
        best_score = 0.0
        q_tokens = set(q_norm.split())

        for key in self._dict:
            key_norm = self._normalize(key)
            key_tokens = set(key_norm.split())
            if q_norm == key_norm:
                best_key = key
                best_score = 1.0
                break
            overlap = len(q_tokens & key_tokens) / max(1, len(q_tokens))
            if overlap > best_score and overlap >= 0.5:
                best_score = overlap
                best_key = key

        if best_key and best_score >= 0.5:
            entry = self._dict[best_key]
            res = SearchResult(
                query=query,
                success=True,
                definition=entry["definition"],
                category=entry.get("category", ""),
                formula=entry.get("formula"),
                sources_used=["offline_business_dictionary"],
                note=f"Matched term: '{best_key}'",
            )
        else:
            note_bits = ["Term not in built-in inventory/finance dictionary."]
            web_results = []
            if self._web_search_fn is not None:
                try:
                    web_results = self._web_search_fn(query)
                except Exception as exc:
                    note_bits.append(f"Web search failed: {exc}")
            if web_results:
                combined = " ".join(web_results[:3])
                res = SearchResult(
                    query=query,
                    success=True,
                    definition=combined,
                    sources_used=["web_search"],
                    note="; ".join(note_bits) + " Results from web lookup.",
                )
            else:
                note_bits.append("No web results returned.")
                res = SearchResult(
                    query=query,
                    success=False,
                    note="; ".join(note_bits),
                )

        self.history.append(res)
        return res

    @staticmethod
    def detect_definition_need(text: str) -> List[str]:
        pattern = re.compile(r"(?:define|definition of|meaning of|what is|what's|explain)\s+([A-Za-z][A-Za-z0-9 \-/&]{1,50}?)(?:\?|$|\.|\,|\;)", re.IGNORECASE)
        matches = [m.group(1).strip(" ?.") for m in pattern.finditer(text)]
        q = text.strip().rstrip("?").strip()
        if re.match(r"^(what|define|explain)\s+(is\s+)?[A-Z]", q, re.IGNORECASE) and len(q.split()) <= 8:
            tail = re.sub(r"^(what|define|explain)\s+(is\s+|are\s+)?(a\s+|an\s+|the\s+)?", "", q, flags=re.IGNORECASE)
            tail = tail.strip(" ?.")
            if 2 < len(tail) < 60 and tail.lower() not in [m.lower() for m in matches]:
                matches.append(tail)
        return [m for m in matches if 2 <= len(m) <= 60]
