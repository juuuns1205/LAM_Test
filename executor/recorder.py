import json
import time
import os
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


DATA_DIR = Path("data")
ACTION_LOG_PATH = DATA_DIR / "actions.json"


def inject_js_recorder(driver):
    """브라우저에 JS 코드 주입해서 클릭/타이핑 기록"""
    driver.execute_script("""
    window.recorded = [];

    function getXPath(element) {
        if (element.id !== '') {
            return "//*[@id='" + element.id + "']";
        }
        if (element === document.body) {
            return '/html/' + element.tagName.toLowerCase();
        }

        let ix = 0;
        let siblings = element.parentNode ? element.parentNode.childNodes : [];
        for (let i=0; i<siblings.length; i++) {
            let sibling = siblings[i];
            if (sibling === element) {
                return getXPath(element.parentNode) + '/' + element.tagName.toLowerCase() + '[' + (ix+1) + ']';
            }
            if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                ix++;
            }
        }
    }

    document.addEventListener('click', e => {
      let cssSel = e.target.tagName;
      if (e.target.id) cssSel += '#' + e.target.id;
      if (e.target.name) cssSel += "[name='" + e.target.name + "']";

      window.recorded.push({
        action: 'click',
        selector: {css: cssSel, xpath: getXPath(e.target)},
        value: null,
        timestamp: Date.now()
      });
    });

    document.addEventListener('input', e => {
      let cssSel = e.target.tagName;
      if (e.target.id) cssSel += '#' + e.target.id;
      if (e.target.name) cssSel += "[name='" + e.target.name + "']";

      window.recorded.push({
        action: 'input',
        selector: {css: cssSel, xpath: getXPath(e.target)},
        value: e.target.value,
        timestamp: Date.now()
      });
    });
    """)
    print("✅ Recorder JS injected.")


def save_actions(driver):
    """브라우저에서 기록된 이벤트 가져와서 actions.json 저장"""
    actions = driver.execute_script("return window.recorded || []")
    if not actions:
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    # 기존 로그 불러오기
    if ACTION_LOG_PATH.exists():
        try:
            with open(ACTION_LOG_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = []
    else:
        existing = []

    # 병합 후 저장
    all_actions = existing + actions
    with open(ACTION_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(all_actions, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved {len(actions)} actions, total = {len(all_actions)}")

    # 배열 초기화 (중복 방지)
    driver.execute_script("window.recorded = []")


def main():
    chrome_options = Options()
    # chrome_options.add_argument("--headless=new")  # 필요하면 headless
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get("https://www.naver.com")  # 원하는 사이트로 변경
        inject_js_recorder(driver)

        print("👉 브라우저에서 타이핑/클릭을 해보세요. (Ctrl+C로 종료)")
        while True:
            save_actions(driver)
            time.sleep(2)
    except KeyboardInterrupt:
        print("🛑 Recording stopped.")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
