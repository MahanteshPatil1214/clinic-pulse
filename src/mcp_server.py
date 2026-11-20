"""FastMCP server exposing the four read-only Clinic Pulse tools."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

try:
    import tools
except ImportError:
    from src import tools

mcp = FastMCP("clinic-pulse")


@mcp.tool()
def get_at_risk_clinics(rep_id: str, top_n: int = 10) -> dict:
    """Rank the clinics most at risk of reducing orders for one rep."""
    return tools.get_at_risk_clinics(rep_id, top_n)


@mcp.tool()
def get_clinic_history(clinic_id: str) -> dict:
    """Return twelve months of synthetic orders and visits for a clinic."""
    return tools.get_clinic_history(clinic_id)


@mcp.tool()
def explain_risk(clinic_id: str) -> dict:
    """Return the top three SHAP reasons for a clinic's risk score."""
    return tools.explain_risk(clinic_id)


@mcp.tool()
def search_policy(query: str, top_k: int = 3) -> dict:
    """Search the synthetic policy documents and return cited chunks."""
    return tools.search_policy(query, top_k)


if __name__ == "__main__":
    mcp.run()
