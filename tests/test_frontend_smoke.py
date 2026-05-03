from __future__ import annotations

import pytest


def test_frontend_run_all_smoke(live_server):
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(live_server, wait_until="networkidle")
        page.get_by_role("button", name="Run All").click()
        page.locator("#run-all-feedback").wait_for(state="visible", timeout=5000)
        page.wait_for_function(
            "() => document.querySelector('#run-all-feedback')?.textContent.includes('Run All complete')",
            timeout=45000,
        )
        page.get_by_role("button", name="Cyber Defence").click()
        page.wait_for_selector("#evidence-grid .evidence-card", timeout=15000)
        page.get_by_role("button", name="Charts", exact=True).click()
        page.wait_for_selector("#chart-explorer .main-svg, #chart-explorer .chart-fallback", timeout=20000)
        browser.close()
