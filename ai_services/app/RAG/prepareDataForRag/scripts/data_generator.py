
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup, Comment
import json
import time
import os
import re
from dotenv import load_dotenv
from ddgs import DDGS

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemma-3-27b-it" 

if not API_KEY:
    print(" Lỗi: Chưa tìm thấy API KEY trong file .env")
    exit()

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    print(f" Lỗi cấu hình Gemini: {e}")
    exit()

DB_FILE = os.path.join(os.path.dirname(__file__), "../smartchef_dataset.json")

# === HARDCODED MENU LIST (150+ Món) ===
MENU_LIST = [
    # --- Món Gà ---
    "Gà kho gừng", "Gà kho sả ớt", "Gà chiên nước mắm", "Gà luộc lá chanh", "Gà xào sả ớt",
    "Gà nấu nấm", "Canh gà lá giang", "Gà rang muối", "Gà hấp hành", "Gà nướng mật ong",
    "Cánh gà chiên bơ", "Cháo gà", "Gỏi gà bắp cải", "Cơm gà hải nam", "Gà hầm thuốc bắc",
    "Gà xào chua ngọt", "Gà xào lăn", "Gà kho nghệ", "Gà rang gừng", "Súp gà ngô non",

    # --- Món Heo ---
    "Thịt kho tàu", "Thịt kho tiêu", "Thịt kho mắm ruốc", "Canh sườn hầm rau củ", "Sườn xào chua ngọt",
    "Thịt luộc cà pháo", "Thịt ba chỉ cháy cạnh", "Thịt băm rang", "Chả lá lốt", "Thịt đông",
    "Canh bí đao thịt bằm", "Canh khổ qua nhồi thịt", "Móng giò hầm măng", "Nem rán", "Chả giò",
    "Bún chả", "Bún thịt nướng", "Thịt heo xào giá hẹ", "Thịt heo quay", "Sườn nướng tảng",
    "Canh mồng tơi thịt bằm", "Canh rau ngót thịt bằm", "Đậu hũ nhồi thịt sốt cà", "Bắp cải cuộn thịt",

    # --- Món Bò ---
    "Bò kho", "Bò lúc lắc", "Bò xào hành tây", "Bò xào cần tỏi", "Bò sốt vang",
    "Bò bít tết", "Canh kim chi thịt bò", "Bò xào bông cải", "Phở bò", "Bún bò huế",
    "Thịt bò xào rau muống", "Bò nướng lá lốt", "Bò nhúng dấm", "Gỏi bò bóp thấu", "Bò hầm khoai tây",
    "Bò xào ớt chuông", "Bò xào đậu bắp", "Bò tái chanh", "Bò cuộn nấm kim châm", "Mì xào bò",

    # --- Món Cá & Hải Sản ---
    "Cá kho tộ", "Cá chép om dưa", "Canh chua cá lóc", "Cá diêu hồng hấp xì dầu", "Cá chiên xù",
    "Cá nục kho cà", "Cá hú kho tộ", "Tôm rim thịt", "Tôm rang me", "Tôm hấp bia",
    "Tôm xào thập cẩm", "Canh bầu nấu tôm", "Mực xào cần tỏi", "Mực nhồi thịt", "Mực chiên mắm",
    "Chả cá thác lác", "Canh riêu cua", "Cua rang me", "Gỏi tôm thịt", "Bún riêu cua",
    "Canh ngao nấu chua", "Hến xúc bánh đa", "Ốc hương xào bơ tỏi", "Cá kèo kho rau răm",

    # --- Món Trứng & Đậu ---
    "Trứng chiên hành", "Trứng chiên thịt bằm", "Trứng luộc", "Trứng ốp la", "Trứng cút lộn xào me",
    "Đậu hũ sốt cà chua", "Đậu hũ chiên sả ớt", "Đậu hũ nhồi thịt", "Khổ qua xào trứng", "Canh đậu hũ hẹ",

    # --- Món Rau & Canh (Chay/Mặn) ---
    "Rau muống xào tỏi", "Rau lang xào tỏi", "Su su xào tỏi", "Cải thìa xào dầu hào", "Bắp cải luộc",
    "Đậu bắp luộc", "Canh rau dền tôm khô", "Canh mướp mồng tơi", "Canh chua chay", "Nấm kho tiêu",
    "Cà tím nướng mỡ hành", "Bí đỏ xào tỏi", "Giá hẹ xào đậu hũ", "Salad trộn dầu giấm", "Nộm hoa chuối",
    "Gỏi ngó sen tôm thịt", "Dưa chua xào lòng", "Canh khoai mỡ", "Canh khoai sọ sườn heo",

    # --- Món Ăn Sáng/Vặt ---
    "Bánh mì ốp la", "Xôi gà", "Xôi gấc", "Bánh cuốn", "Bún riêu",
    "Mì Quảng", "Hủ tiếu nam vang", "Bánh canh cua", "Nui xào bò", "Súp cua",
    "Cháo lòng", "Cháo trai", "Khoai tây chiên", "Khoai lang kén", "Ngô chiên bơ"
]

def find_dmx_links(dish_name):
    query = f"cách làm {dish_name} site:dienmayxanh.com"
    links = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region="vn-vi", max_results=5))
        
        for r in results:
            link = r.get("href", "")
            if "dienmayxanh.com" in link:
                links.append(link)
    except Exception as e:
        print(f" Lỗi DuckDuckGo: {e}")
    return links

def get_html_strict(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.content, 'html.parser')
        
        content_parts = []
        
        # Chỉ lấy khi có ĐỦ cả 2 class chuẩn
        staple_div = soup.select_one('div.staple')
        method_div = soup.select_one('div.method')

        if staple_div and method_div:
            # Clean up method
            for tag in method_div.select('.box-gallery, .tipsrecipe, .note, script, style'):
                tag.decompose()
            
            content_parts.append(staple_div.get_text(separator=' ', strip=True))
            content_parts.append(method_div.get_text(separator='\n', strip=True))
            full_content = "\n\n".join(content_parts)
            return full_content[:30000]

        return None
    except Exception:
        return None

def process_to_json(html_text, original_name):
    
    prompt = f"""
    Bạn là chuyên gia dữ liệu ẩm thực Việt Nam. Nhiệm vụ: Trích xuất công thức từ văn bản raw bên dưới thành JSON chuẩn.
    
    TÊN MÓN GỐC: "{original_name}"
    
    YÊU CẦU QUAN TRỌNG (STRICT):
    1. **nguyen_lieu_chinh**: Chỉ liệt kê các thành phần CẤU TRÚC món ăn (Thịt, Cá, Rau, Củ, Đậu, Trứng, Bún, Phở...). 
       - CHUẨN HÓA tên gọi: "thịt ba chỉ" -> "thịt heo", "cá lóc đồng" -> "cá lóc", "trứng gà ta" -> "trứng".
       - KHÔNG đưa gia vị vào đây.
    2. **gia_vi**: Liệt kê riêng các loại gia vị, rau thơm, đồ nêm nếm (Nước mắm, Muối, Đường, Tiêu, Tỏi, Ớt, Hành tím, Gừng, Dầu ăn...).
    3. **nguyen_lieu_search**: Chuỗi các từ khóa của `nguyen_lieu_chinh` (đã chuẩn hóa), viết thường, cách nhau dấu phẩy. Dùng để search vector.
    4. **cach_lam:** Giữ nguyên cách làm từ trong html (các bước thực hiện), chỉ refactor thành bước 1, bước 2.....

    VĂN BẢN NGUỒN:
    '''{html_text}'''

    JSON OUTPUT FORMAT (Trả về JSON thuần, không markdown):
    {{
        "id": "slug-khong-dau",
        "ten_mon": "Tên món chính xác từ bài viết",
        "mo_ta": "Mô tả ngắn 1 câu hấp dẫn",
        "nguyen_lieu_chinh": ["thịt gà", "gừng"], 
        "gia_vi": ["nước mắm", "muối", "tiêu", "dầu ăn"],
        "nguyen_lieu_search": "thịt gà, gừng", 
        "nguyen_lieu_chi_tiet": ["500g thịt gà ta", "1 nhánh gừng", "2 thìa nước mắm"],
        "cach_lam": ["Bước 1...", "Bước 2..."],
        "thoi_gian_nau": "30 phút"
    }}
    Nếu không tìm thấy công thức, trả về: {{}}
    """
    
    max_retries = 5
    wait_time = 20

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:-3].strip()
            elif text.startswith("```"):
                text = text[3:-3].strip()
                
            data = json.loads(text)
            print(data)
            return data
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "ResourceExhausted" in error_msg:
                print(f" Hết quota (429). Đợi {wait_time}s rồi thử lại... (Lần {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                wait_time *= 1.5 
            else:
                print(f" Lỗi Parse/Gen AI: {e}")
                return None
    
    print(" Bỏ qua món này sau nhiều lần retry thất bại.")
    return None

def save_append(data):
    current_data = []
    # Đọc data cũ
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                current_data = json.loads(f.read())
        except: pass

    # Check trùng
    for item in current_data:
        if item.get('id') == data.get('id'):
            print(f" Đã có: {data.get('ten_mon')}")
            return

    # Lưu mới 
    current_data.append(data)
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(current_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Đã lưu: {data.get('ten_mon')} (Tổng: {len(current_data)})")

import unicodedata

def slugify(value):

    value = str(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-')

def load_existing_ids():
    if not os.path.exists(DB_FILE):
        return set()
    try:
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(item.get('id') for item in data)
    except:
        return set()

def main():
    dish_list_path = os.path.join(os.path.dirname(__file__), "scripts/dish_list.txt") 
    if not os.path.exists(dish_list_path):
        dish_list_path = os.path.join(os.path.dirname(__file__), "dish_list.txt")

    if not os.path.exists(dish_list_path):
        print(f"Không tìm thấy file danh sách món: {dish_list_path}")
        print("Hãy chạy script 'crawler_discovery.py' trước để tạo danh sách.")
        return

    with open(dish_list_path, 'r', encoding='utf-8') as f:
        dishes = [line.strip() for line in f if line.strip()]

    print(f"Tải {len(dishes)} món từ danh sách.")
    
    existing_ids = load_existing_ids()
    print(f" Database hiện có {len(existing_ids)} món. Chế độ: Incremental Update (Chỉ thêm mới).")

    for i, dish in enumerate(dishes):
        print(f"\n--- [{i+1}/{len(dishes)}] Crawling: {dish} ---")
        
        links = find_dmx_links(dish)
        if not links:
            print(" Không tìm thấy link nào trên DuckDuckGo.")
            continue
        
        found_valid_content = False
        
        for link in links:
            html = get_html_strict(link)
            
            if html:
                print(f"Link phù hợp: {link}")
                
                data = process_to_json(html, dish)
                
                if data and data.get("ten_mon"):
                    parsed_name = data.get("ten_mon")
                    final_id = slugify(parsed_name)
                    data['id'] = final_id
                    
                    if final_id in existing_ids:
                        print(f" Bỏ qua (Đã có trong DB): {parsed_name} [{final_id}]")
                        continue

                    save_append(data)
                    existing_ids.add(final_id)
                    found_valid_content = True
                    break 
            else:
                pass 
        
        if not found_valid_content:
            print(f" Không lấy được nội dung hoặc bị trùng cho '{dish}'.")
            
        time.sleep(15)

if __name__ == "__main__":
    main()