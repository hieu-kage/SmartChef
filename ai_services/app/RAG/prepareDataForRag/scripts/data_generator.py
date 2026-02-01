

import google.generativeai as genai
import requests
from bs4 import BeautifulSoup, Comment
import json
import time
import os
import re
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME="gemma-3-27b-it"
DB_FILE=os.getenv("DB_FILE")
MENU_LIST = [
  "Gà kho gừng",
  "Gà kho nghệ",
  "Gà xào hành tây",
  "Gà xào súp lơ",
  "Gà xào nấm",
  "Gà xào ớt chuông",
  "Gà xào dứa",
  "Gà chiên nước mắm",
  "Canh gà nấu nấm",
  "Canh gà hầm củ quả",
  "Cháo gà",
  "Gỏi gà bắp cải",
  "Gỏi gà hành tây",
  "Cơm gà",
  "Thịt heo kho tộ",
  "Thịt heo kho tiêu",
  "Thịt heo kho gừng",
  "Thịt heo xào hành tây",
  "Thịt heo xào ớt chuông",
  "Thịt heo xào đậu que",
  "Thịt heo xào nấm",
  "Thịt heo băm sốt cà chua",
  "Sườn xào chua ngọt",
  "Canh bí đỏ thịt bằm",
  "Canh bắp cải thịt bằm",
  "Canh súp lơ thịt bằm",
  "Canh khoai tây sườn heo",
  "Canh củ cải sườn heo",
  "Cháo thịt bằm",
  "Bắp cải cuộn thịt",
  "Cà tím nhồi thịt",
  "Đậu bắp nhồi thịt",
  "Bí ngòi xào thịt heo",
  "Bò xào hành tây",
  "Bò xào ớt chuông",
  "Bò xào bông cải xanh",
  "Bò xào nấm",
  "Bò xào dứa",
  "Bò xào tỏi",
  "Bò xào đậu bắp",
  "Bò kho",
  "Bò lúc lắc",
  "Bò hầm khoai tây",
  "Canh bò hầm bí đỏ",
  "Cháo bò",
  "Salad bò",
  "Tôm rim",
  "Tôm rang tỏi",
  "Tôm xào thập cẩm",
  "Tôm xào bông cải",
  "Tôm xào nấm",
  "Tôm xào ớt chuông",
  "Tôm xào dứa",
  "Tôm sốt cà chua",
  "Canh chua tôm",
  "Canh bí đỏ nấu tôm",
  "Canh bắp cải nấu tôm",
  "Cháo tôm",
  "Gỏi tôm xoài",
  "Gỏi tôm đu đủ",
  "Cá kho gừng",
  "Cá kho nghệ",
  "Cá chiên sốt cà chua",
  "Cá hấp hành gừng",
  "Canh chua cá",
  "Canh cá nấu ngót",
  "Cháo cá",
  "Bắp cải xào tỏi",
  "Bắp cải luộc",
  "Súp lơ xào tỏi",
  "Súp lơ luộc",
  "Đậu bắp luộc",
  "Đậu bắp xào tỏi",
  "Cà tím nướng mỡ hành",
  "Cà tím xào tỏi",
  "Cà tím bung",
  "Bí đỏ xào tỏi",
  "Bí ngòi xào tỏi",
  "Nấm xào tỏi",
  "Khoai tây chiên",
  "Khoai tây xào tỏi",
  "Salad trộn dầu giấm",
  "Salad dưa chuột cà chua",
  "Gỏi dưa chuột",
  "Gỏi đu đủ",
  "Gỏi xoài",
  "Cơm chiên thập cẩm",
  "Cơm bò xào",
  "Sinh tố bơ",
  "Sinh tố xoài",
  "Sinh tố dâu tây",
  "Sinh tố chuối",
  "Nước ép cà rốt",
  "Nước ép dứa",
  "Nước chanh",
  "Thịt heo luộc",
  "Canh bí đỏ chay",
  "Cà rốt xào thịt bò",
  "Nấm xào thịt bò",
  "Đậu que xào thịt bò"
]
if not API_KEY:
    print(" Lỗi: Chưa tìm thấy API KEY trong file .env hoặc biến môi trường.")
    exit()

genai.configure(api_key=API_KEY)


model = genai.GenerativeModel(MODEL_NAME)


from ddgs import DDGS

def find_dmx_link(dish_name):
    print(f" Đang tìm link cho món: {dish_name}...")

    query = f"cách làm {dish_name} site:dienmayxanh.com"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                region="vn-vi",
                max_results=10
            ))

        print(f"   🔍 Tìm thấy {len(results)} kết quả:")
        for i, r in enumerate(results):
            print(f"      [{i+1}] {r.get('href')}")

        for r in results:
            link = r.get("href", "")
            if "dienmayxanh.com" in link:
                print(f"    Đã chọn trang: {link}")
                return link

        print(f"    Không có link DienMayXanh trong kết quả.")

    except Exception as e:
        print(f"   Lỗi DuckDuckGo ({query}): {e}")

    print("   ->  Không tìm thấy link nào khả thi.")
    return None

def get_html_special(url):
    print(f"⬇Đang tải HTML: {url}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.content, 'html.parser')

        content_html = ""
        target_comment = None

        for element in soup.find_all(string=lambda text: isinstance(text, Comment)):
            if "region Web page" in element:
                target_comment = element
                break

        if target_comment:
            parent = target_comment.parent
            if parent:
                for tag in parent(["script", "style", "iframe", "div.comment-box", "div.box-comment"]):
                    tag.decompose()
                content_html = parent.get_text(separator='\n', strip=True)
            else:
                prev_tag = target_comment.find_previous_sibling()
                if prev_tag:
                    content_html = prev_tag.get_text(separator='\n', strip=True)

        if not content_html or len(content_html) < 200:
            print("    Logic Comment không hiệu quả, dùng logic class chuẩn DMX.")
            article = soup.select_one('.news-content') or soup.select_one('article') or soup.select_one('.box-content')
            if article:
                for tag in article(["script", "style", "div.relate-news", "div.comment-box"]):
                    tag.decompose()
                content_html = article.get_text(separator='\n', strip=True)
            else:
                content_html = soup.body.get_text(separator='\n', strip=True)

        return content_html[:30000]

    except Exception as e:
        print(f"   -> Lỗi tải trang: {e}")
        return None


def process_to_json(html_text, original_name):
    print(f" Đang nhờ AI trích xuất JSON cho món: {original_name}...")
    prompt = f"""
    Bạn là chuyên gia dữ liệu. Nhiệm vụ: Trích xuất công thức nấu ăn từ văn bản hỗn độn dưới đây thành JSON chuẩn.
    Tên món gốc dự kiến: "{original_name}"

    VĂN BẢN NGUỒN:
    '''{html_text}'''

    YÊU CẦU OUTPUT:
    1. Trả về DUY NHẤT 1 JSON Object hợp lệ (không markdown, không giải thích).
    2. Format JSON:
    {{
        "id": "slug-khong-dau-cach-noi-bang-gach-ngang",
        "ten_mon": "Tên chính xác trong bài viết (nếu khác tên gốc)",
        "mo_ta": "Mô tả ngắn gọn về món ăn (1 câu)",
        "nguyen_lieu_search": "liệt kê nguyên liệu chính, viết thường, ngăn cách phẩy (để search)",
        "nguyen_lieu_chi_tiet": ["500g thịt gà", "1 củ gừng", "gia vị..."],
        "cach_lam": ["Bước 1: Sơ chế...", "Bước 2: Nấu..."],
        "thoi_gian_nau": "Ước lượng (VD: 30 phút)"
    }}
    3. Nếu nội dung input lỗi hoặc không phải bài công thức, trả về JSON rỗng: {{}}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()

        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            clean_json = json_match.group()
            return json.loads(clean_json)
        else:
            return json.loads(text.replace('```json', '').replace('```', ''))

    except Exception as e:
        print(f"   -> Lỗi Parse JSON: {e}")
        return None


def save_append(data):
    current_data = []
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
                if content:
                    current_data = json.loads(content)
        except Exception as e:
            print(f"   -> Lỗi đọc file cũ: {e}, tạo file mới.")

    exists = any(item.get('id') == data.get('id') for item in current_data)
    if not exists:
        current_data.append(data)
        with open(DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, ensure_ascii=False, indent=2)
        print(f" Đã lưu món: {data.get('ten_mon')} (Tổng: {len(current_data)})")
    else:
        print(f" Món {data.get('ten_mon')} đã có, bỏ qua.")


def main():
    menu_list = MENU_LIST

    print(f"Có sẵn {len(menu_list)} món để crawl.")

    print(f"Đã lên được {len(menu_list)} món.")

    for i, dish in enumerate(menu_list):
        print(f"\n--- Món {i + 1}/{len(menu_list)}: {dish} ---")

        url = find_dmx_link(dish)
        if not url:
            print("   -> Bỏ qua món này.")
            time.sleep(2)
            continue

        html_content = get_html_special(url)
        if not html_content:
            continue

        json_data = process_to_json(html_content, dish)

        if json_data and json_data.get("ten_mon"):
            save_append(json_data)
        else:
            print("   -> Dữ liệu trích xuất rỗng hoặc lỗi.")

        print("💤 Nghỉ 5s...")
        time.sleep(5)

if __name__ == "__main__":
    main()