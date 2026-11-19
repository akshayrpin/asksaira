"""
backend/admin — the staff console's server side (the only authed surface, mounted under /admin).

Analytics reads (insights_data over query_traces) today; source/route/canonical config writes next.
All Cosmos access is borrowed from backend.observability so there is one client/credential owner.
"""
