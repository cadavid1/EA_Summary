import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from datetime import datetime
from multiprocessing import freeze_support

# Set up headers to mimic a real browser.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
}

# Global cache dictionary for executive order texts.
order_text_cache = {}

def get_summarizer():
    """
    Initialize the summarization pipeline.
    This is called within main() to avoid multiprocessing issues.
    """
    st.write("Initializing summarizer (CPU mode)...")
    return pipeline("summarization", model="allenai/led-base-16384", device=-1)

def fetch_order_text(url):
    """
    Fetch the detailed text of an executive order from its detail page.
    Checks cache to avoid re-downloading the same page.
    """
    if url in order_text_cache:
        st.write(f"Using cached order text for: {url}")
        return order_text_cache[url]
    
    try:
        st.write(f"Fetching detail page: {url}")
        detail_response = requests.get(url, headers=HEADERS)
        if detail_response.status_code != 200:
            st.write(f"Failed to fetch detail page: {url}")
            return ""
        detail_soup = BeautifulSoup(detail_response.content, "html.parser")
        
        selectors = [
            ("article", {}),
            ("div", {"class": "page-content"}),
            ("main", {}),
            ("div", {"class": "content-wrapper"}),
            ("div", {"class": "page__content"}),
            ("div", {"class": "entry-content"})
        ]
        order_text = ""
        for tag, attrs in selectors:
            container = detail_soup.find(tag, attrs=attrs)
            if container:
                order_text = container.get_text(separator=" ", strip=True)
                if len(order_text) > 200:
                    break
        if not order_text or len(order_text) < 200:
            order_text = detail_soup.get_text(separator=" ", strip=True)
        
        # Cache the fetched order text.
        order_text_cache[url] = order_text
        return order_text
    except Exception as e:
        st.write(f"Exception in fetch_order_text: {e}")
        return ""

def fetch_executive_orders(summarizer):
    """
    Scrape executive orders from the White House website.
    """
    base_url = "https://www.whitehouse.gov"
    orders = []
    page = 1

    while True:
        page_url = f"{base_url}/presidential-actions/" if page == 1 else f"{base_url}/presidential-actions/page/{page}/"
        st.write(f"Fetching page: {page_url}")
        response = requests.get(page_url, headers=HEADERS)
        if response.status_code != 200:
            st.write(f"Failed to fetch {page_url}, status: {response.status_code}")
            break
        
        soup = BeautifulSoup(response.content, "html.parser")
        items = soup.select("li[data-wp-key^='post-template-item-']")
        if not items:
            st.write(f"No items found on page {page}. Ending fetch.")
            break

        page_orders = []
        for item in items:
            h2 = item.find("h2", class_="wp-block-post-title")
            if not h2:
                st.write("No <h2> found in item; skipping.")
                continue
            a_tag = h2.find("a", href=True)
            if not a_tag:
                st.write("No <a> tag found in <h2>; skipping.")
                continue

            title = a_tag.get_text(strip=True)
            link = a_tag["href"]

            time_tag = item.find("time")
            if time_tag and time_tag.has_attr("datetime"):
                try:
                    order_date = datetime.fromisoformat(time_tag["datetime"]).date()
                except Exception as e:
                    st.write(f"Error parsing date for {title}: {e}")
                    continue
            else:
                st.write(f"No time element found in item for {title}; skipping.")
                continue

            st.write(f"Processing: {title} dated {order_date}")
            order_text = fetch_order_text(link)
            if order_text:
                try:
                    summary_output = summarizer(order_text, max_length=130, min_length=30, do_sample=False)
                    summary = summary_output[0]['summary_text']
                except Exception as e:
                    st.write(f"Summarization error for {title}: {e}")
                    summary = "Summary generation error."
            else:
                summary = ""

            impact = "High" if "emergency" in order_text.lower() else "Moderate"
            page_orders.append({
                "Date": order_date,
                "Title": title,
                "Link": link,
                "Summary": summary,
                "Impact": impact,
                "Full Text": order_text
            })

        if not page_orders:
            st.write(f"No orders found on page {page}. Ending fetch.")
            break

        orders.extend(page_orders)
        page += 1

    return orders

def main():
    # Inject meta refresh tag to automatically reload the page every 3600 seconds (1 hour)
    components.html("<meta http-equiv='refresh' content='3600'>", height=0)

    st.title("Executive Orders Dashboard with LED Summarization")
    st.write("Fetching executive orders and generating summaries...")

    # Initialize the summarizer inside main() to avoid multiprocessing issues.
    summarizer = get_summarizer()
    orders = fetch_executive_orders(summarizer)
    df = pd.DataFrame(orders)

    if df.empty:
        st.write("No executive orders found.")
    else:
        date_filter = st.date_input("Show orders from:", value=datetime(2025, 1, 20).date())
        df_filtered = df[df["Date"] >= date_filter]
        st.write("Executive Orders Overview:")
        st.dataframe(df_filtered[["Date", "Title", "Link", "Summary", "Impact"]])

if __name__ == "__main__":
    freeze_support()  # Ensure proper multiprocessing support on macOS.
    main()
