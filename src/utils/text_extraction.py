"""
Text extraction utilities for scraping content from HuggingFace model pages.
"""

import re
import time
import random
import requests
from bs4 import BeautifulSoup
from typing import Optional


def extract_text_from_huggingface(url: str, selector: str = '.model-card-content') -> str:
    """
    Extract and clean text from Hugging Face pages by emulating a browser.
    
    Args:
        url (str): The URL of the Hugging Face page to scrape
        selector (str): CSS selector to target specific content (default: '.model-card-content')
        
    Returns:
        str: Cleaned text content with proper spacing and formatting
    """
    try:
        # Set up advanced browser-like headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Sec-Ch-Ua': '"Google Chrome";v="120", "Chromium";v="120", "Not-A.Brand";v="99"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Cache-Control': 'max-age=0',
            'Priority': 'u=0, i',
            'DNT': '1',
        }
        
        # Create a session to maintain cookies
        session = requests.Session()
        
        # Add a referer if it's not the main page
        if 'huggingface.co' in url and not url.endswith('huggingface.co'):
            headers['Referer'] = 'https://huggingface.co/'
        
        # First visit the main page to get cookies
        session.get('https://huggingface.co/', headers=headers)
        
        # Add a small delay to mimic human browsing (optional)
        time.sleep(random.uniform(1, 2))
        
        # Now request the actual page
        response = session.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the targeted content
        content_div = soup.select_one(selector)
        if not content_div:
            # Try a broader approach if selector not found
            content_div = soup.select_one('main') or soup.body
            if not content_div:
                return "Target content not found. Please check the page structure."
        
        # Process and clean HTML content
        
        # Process text within different HTML elements
        
        # Handle paragraph spacing
        for p in content_div.find_all('p'):
            if p.string:
                p.string.replace_with(p.string + '\n\n')
        
        # Handle headings with importance level
        for i in range(1, 7):
            for heading in content_div.find_all(f'h{i}'):
                if heading.string:
                    heading.string.replace_with(heading.string + '\n\n')
        
        # Handle list items
        for li in content_div.find_all('li'):
            if li.string:
                li.string.replace_with("• " + li.string + '\n')
        
        # Handle code blocks
        for code in content_div.find_all(['pre', 'code']):
            if code.string:
                code.string.replace_with("\n```\n" + code.string + "\n```\n")
        
        # Handle tables
        for table in content_div.find_all('table'):
            # Create a placeholder to indicate table content
            table_text = "\n--- Table Content ---\n"
            
            # Process table headers
            headers = []
            for th in table.find_all('th'):
                headers.append(th.get_text(strip=True))
            
            if headers:
                table_text += " | ".join(headers) + "\n"
                table_text += "-" * (len(table_text) - 1) + "\n"
            
            # Process table rows
            for tr in table.find_all('tr'):
                row_data = []
                for td in tr.find_all('td'):
                    row_data.append(td.get_text(strip=True))
                if row_data:
                    table_text += " | ".join(row_data) + "\n"
            
            table_text += "--- End Table ---\n\n"
            table.replace_with(soup.new_string(table_text))
        
        # Handle divs that contain text directly
        for div in content_div.find_all('div'):
            if div.string and not div.find_all():  # Only direct text, no children
                div.string.replace_with(div.string + '\n')
        
        # Extract text with better spacing
        text = content_div.get_text(separator='\n', strip=False)
        
        # Clean up the text more carefully
        # First, normalize whitespace but preserve structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Clean each line individually
            line = line.strip()
            if line:  # Only add non-empty lines
                # Fix multiple spaces within a line
                line = re.sub(r'\s+', ' ', line)
                cleaned_lines.append(line)
        
        # Join lines with proper spacing
        text = '\n'.join(cleaned_lines)
        
        # Fix spacing after periods, question marks, and exclamation points
        text = re.sub(r'([.!?])(?=[A-Z])', r'\1 ', text)
        
        # Remove excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
        
    except requests.exceptions.RequestException as e:
        error_message = f"Error fetching the URL: {e}"
        if hasattr(e, 'response') and e.response is not None:
            error_message += f" (Status code: {e.response.status_code})"
            
            # For Hugging Face specific errors
            if e.response.status_code == 401:
                error_message += "\nHugging Face may require a login for this content."
            elif e.response.status_code == 403:
                error_message += "\nAccess forbidden. Hugging Face may be blocking scraping attempts."
            elif e.response.status_code == 429:
                error_message += "\nToo many requests. Try again later or reduce request frequency."
                
        return error_message
    except Exception as e:
        return f"An error occurred: {e}" 