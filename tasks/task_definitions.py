"""
Task definitions for the SQL Query Optimizer environment.
Each task has: schema, slow_query, expected logic, sample_data, hints (easy only).
"""
from typing import Dict, Any

TASKS: Dict[str, Dict[str, Any]] = {

    # ─────────────────────────────────────────────────────────────────
    # EASY: SELECT with redundant subquery and SELECT *
    # ─────────────────────────────────────────────────────────────────
    "easy_select_optimization": {
        "difficulty": "easy",
        "task_description": (
            "Return the full name and email of all customers who placed at least one order "
            "in the last 30 days. Order results by last name ascending."
        ),
        "schema_ddl": """
CREATE TABLE customers (
    id          INTEGER PRIMARY KEY,
    first_name  TEXT NOT NULL,
    last_name   TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    total       NUMERIC(10,2) NOT NULL,
    placed_at   TIMESTAMP NOT NULL
);
""",
        "slow_query": """
SELECT * FROM customers
WHERE id IN (
    SELECT customer_id FROM (
        SELECT * FROM orders
    ) AS all_orders
    WHERE all_orders.placed_at >= CURRENT_DATE - INTERVAL '30 days'
);
""",
        "expected_logic": "JOIN or EXISTS/IN on orders.placed_at >= now()-30d, select only name+email, ORDER BY last_name",
        "sample_data": {
            "customers": [
                {"id": 1, "first_name": "Alice", "last_name": "Smith", "email": "alice@example.com"},
                {"id": 2, "first_name": "Bob",   "last_name": "Jones", "email": "bob@example.com"},
                {"id": 3, "first_name": "Carol",  "last_name": "Adams", "email": "carol@example.com"},
            ],
            "orders": [
                {"id": 1, "customer_id": 1, "total": 99.99,  "placed_at": "2025-03-25 10:00:00"},
                {"id": 2, "customer_id": 2, "total": 149.50, "placed_at": "2024-01-01 08:00:00"},
            ]
        },
        "hints": [
            "Use a JOIN instead of a nested subquery",
            "Select only first_name, last_name, email — not SELECT *",
            "Use DISTINCT to avoid duplicate customers",
            "Add ORDER BY last_name ASC",
        ],
        "anti_patterns": ["SELECT *", "SELECT * FROM orders", "nested subquery on orders"],
        "required_clauses": ["ORDER BY", "last_name"],
    },

    # ─────────────────────────────────────────────────────────────────
    # MEDIUM: N+1 style correlated subquery → proper JOIN + aggregation
    # ─────────────────────────────────────────────────────────────────
    "medium_join_optimization": {
        "difficulty": "medium",
        "task_description": (
            "For each product category, return the category name, total revenue "
            "(sum of quantity * unit_price for completed orders), and the number of distinct "
            "products sold. Only include categories with total revenue above $500. "
            "Order by total revenue descending."
        ),
        "schema_ddl": """
CREATE TABLE categories (
    id    INTEGER PRIMARY KEY,
    name  TEXT NOT NULL
);

CREATE TABLE products (
    id          INTEGER PRIMARY KEY,
    category_id INTEGER NOT NULL REFERENCES categories(id),
    name        TEXT NOT NULL,
    unit_price  NUMERIC(10,2) NOT NULL
);

CREATE TABLE orders (
    id         INTEGER PRIMARY KEY,
    status     TEXT NOT NULL,   -- 'pending','completed','cancelled'
    placed_at  TIMESTAMP NOT NULL
);

CREATE TABLE order_items (
    id         INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL
);
""",
        "slow_query": """
SELECT
    c.name,
    (SELECT SUM(oi.quantity * p2.unit_price)
     FROM order_items oi
     JOIN products p2 ON oi.product_id = p2.id
     JOIN orders o2   ON oi.order_id   = o2.id
     WHERE p2.category_id = c.id AND o2.status = 'completed') AS total_revenue,
    (SELECT COUNT(DISTINCT oi2.product_id)
     FROM order_items oi2
     JOIN products p3 ON oi2.product_id = p3.id
     JOIN orders o3   ON oi2.order_id   = o3.id
     WHERE p3.category_id = c.id AND o3.status = 'completed') AS distinct_products
FROM categories c
ORDER BY total_revenue DESC;
""",
        "expected_logic": (
            "Single JOIN path: categories → products → order_items → orders(status=completed). "
            "GROUP BY category. HAVING SUM > 500. No correlated subqueries."
        ),
        "sample_data": {
            "categories": [
                {"id": 1, "name": "Electronics"},
                {"id": 2, "name": "Books"},
                {"id": 3, "name": "Clothing"},
            ],
            "products": [
                {"id": 1, "category_id": 1, "name": "Laptop",  "unit_price": 999.99},
                {"id": 2, "category_id": 1, "name": "Phone",   "unit_price": 699.99},
                {"id": 3, "category_id": 2, "name": "Novel",   "unit_price": 14.99},
                {"id": 4, "category_id": 3, "name": "T-Shirt", "unit_price": 29.99},
            ],
            "orders": [
                {"id": 1, "status": "completed", "placed_at": "2025-03-01"},
                {"id": 2, "status": "cancelled", "placed_at": "2025-03-05"},
            ],
            "order_items": [
                {"id": 1, "order_id": 1, "product_id": 1, "quantity": 2},
                {"id": 2, "order_id": 1, "product_id": 3, "quantity": 5},
                {"id": 3, "order_id": 2, "product_id": 4, "quantity": 3},
            ]
        },
        "hints": None,
        "anti_patterns": ["correlated subquery", "SELECT (SELECT"],
        "required_clauses": ["GROUP BY", "HAVING", "ORDER BY"],
    },

    # ─────────────────────────────────────────────────────────────────
    # HARD: Complex multi-table with window functions needed
    # ─────────────────────────────────────────────────────────────────
    "hard_complex_optimization": {
        "difficulty": "hard",
        "task_description": (
            "For each sales representative, compute: their name, total sales amount for "
            "the current calendar year, their rank among all reps by total sales (1 = highest), "
            "and the percentage their sales represent of the top performer's sales. "
            "Include reps with zero sales. Order by rank ascending."
        ),
        "schema_ddl": """
CREATE TABLE reps (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    region     TEXT NOT NULL,
    hired_at   DATE NOT NULL
);

CREATE TABLE deals (
    id          INTEGER PRIMARY KEY,
    rep_id      INTEGER NOT NULL REFERENCES reps(id),
    amount      NUMERIC(12,2) NOT NULL,
    closed_at   DATE NOT NULL,
    status      TEXT NOT NULL   -- 'won','lost','open'
);
""",
        "slow_query": """
SELECT
    r.name,
    (SELECT COALESCE(SUM(d.amount), 0)
     FROM deals d
     WHERE d.rep_id = r.id
       AND d.status = 'won'
       AND EXTRACT(YEAR FROM d.closed_at) = EXTRACT(YEAR FROM CURRENT_DATE)) AS total_sales,
    (SELECT COUNT(*) + 1
     FROM reps r2
     WHERE (SELECT COALESCE(SUM(d2.amount), 0)
            FROM deals d2
            WHERE d2.rep_id = r2.id
              AND d2.status = 'won'
              AND EXTRACT(YEAR FROM d2.closed_at) = EXTRACT(YEAR FROM CURRENT_DATE))
           >
           (SELECT COALESCE(SUM(d3.amount), 0)
            FROM deals d3
            WHERE d3.rep_id = r.id
              AND d3.status = 'won'
              AND EXTRACT(YEAR FROM d3.closed_at) = EXTRACT(YEAR FROM CURRENT_DATE))
    ) AS rank
FROM reps r
ORDER BY rank ASC;
""",
        "expected_logic": (
            "CTE or subquery aggregating won deals for this year per rep. "
            "LEFT JOIN reps to preserve zero-sales reps. "
            "RANK() or DENSE_RANK() window function for ranking. "
            "Percentage = rep_sales / MAX(rep_sales) OVER () * 100. "
            "No correlated subqueries."
        ),
        "sample_data": {
            "reps": [
                {"id": 1, "name": "Diana Prince", "region": "West",  "hired_at": "2020-01-15"},
                {"id": 2, "name": "Bruce Wayne",  "region": "East",  "hired_at": "2019-06-01"},
                {"id": 3, "name": "Clark Kent",   "region": "North", "hired_at": "2021-03-10"},
                {"id": 4, "name": "Lois Lane",    "region": "South", "hired_at": "2022-08-22"},
            ],
            "deals": [
                {"id": 1, "rep_id": 1, "amount": 50000.00, "closed_at": "2025-02-14", "status": "won"},
                {"id": 2, "rep_id": 1, "amount": 30000.00, "closed_at": "2025-03-01", "status": "won"},
                {"id": 3, "rep_id": 2, "amount": 90000.00, "closed_at": "2025-01-20", "status": "won"},
                {"id": 4, "rep_id": 3, "amount": 20000.00, "closed_at": "2024-12-15", "status": "won"},
                {"id": 5, "rep_id": 3, "amount": 15000.00, "closed_at": "2025-02-28", "status": "lost"},
            ]
        },
        "hints": None,
        "anti_patterns": ["triple nested correlated subquery", "rank via COUNT(*)+1 subquery"],
        "required_clauses": ["WITH", "RANK()", "OVER", "LEFT JOIN", "COALESCE"],
    },
}
