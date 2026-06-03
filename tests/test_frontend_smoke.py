from __future__ import annotations

import pytest


def test_apply_updates_experiment_context(live_server):
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(live_server, wait_until="networkidle")
            page.wait_for_function("() => window.ExperimentContext && window.ExperimentContext.current")
            page.locator("#analysis-window-size").fill("100")
            page.locator("#analysis-flow-index").fill("1")
            page.locator("#analysis-alpha").fill("0.55")
            page.locator("#analysis-beta").fill("0.45")
            page.locator("#analysis-fusion-mode").select_option("soft")
            page.locator("#analysis-seed").fill("9")
            page.get_by_role("button", name="Apply", exact=True).click()
            page.wait_for_function(
                """() => {
                    const context = window.ExperimentContext.current();
                    const summary = document.querySelector('#analysis-param-summary')?.textContent || '';
                    return context.window === 100
                        && context.flow === 1
                        && context.alpha === 0.55
                        && context.beta === 0.45
                        && context.fusion === 'soft'
                        && context.seed === 9
                        && summary.includes('window 100')
                        && summary.includes('seed 9');
                }""",
                timeout=15000,
            )
            page.get_by_role("button", name="Charts", exact=True).click()
            page.wait_for_selector("#metric-comparison-caption", timeout=15000)
        finally:
            browser.close()


def test_frontend_run_all_smoke(live_server):
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
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
        finally:
            browser.close()
