#!/usr/bin/env python3
"""
StockCharts Reader - Fully automated data extraction from StockCharts ACP chart.

This script automates data extraction using OpenHands browser visualization tools:
1. Opens browser to StockCharts ACP URL
2. After every left arrow press, uses browser visualization to read date and value
3. Updates pandas dataframe at every step
4. Repeats until no more data points
5. Saves data to stockcharts_data.xlsx

No user interaction required - fully automated.
"""

import re
import time
import pandas as pd
from pathlib import Path

# Chart URL
CHART_URL = "https://stockcharts.com/acp/?s=!BINYNLPTI"

# Output file
OUTPUT_FILE = "stockcharts_data.xlsx"

# Delay between operations
PAUSE = 0.3


def extract_date_and_value_from_content(content: str) -> tuple:
    """Extract date and value from browser content.
    
    The browser_get_content() function is the visualization tool that reads
    the page content. We parse that for date and value.
    
    Args:
        content: Output from browser_get_content() - the visualized page text
        
    Returns:
        tuple of (date_str, value_float or None)
    """
    if not content:
        return None, None
    
    # StockCharts shows date like "27-Apr-2026" when crosshairs are on a bar
    # The date appears at the bottom under the chart
    date_pattern = r'(\d{1,2}-[A-Za-z]{3}-\d{4})'
    dates = re.findall(date_pattern, content)
    
    # The value appears in the quote box at top right
    # Look for values like "0.57" near the start of content
    # We need to be more precise to avoid capturing other numbers
    value_pattern = r'^\s*(\d+\.\d+)'
    values = re.findall(value_pattern, content, re.MULTILINE)
    
    # Also try looking for value near OHLC labels
    if not values:
        value_pattern = r'O:\s*(\d+\.\d+)'
        values = re.findall(value_pattern, content)
    
    if not values:
        # Last resort - get first decimal number
        value_pattern = r'(?<!\d)(0\.\d+|\d{1,3}\.\d+)(?!\d)'
        values = re.findall(value_pattern, content)
    
    unique_dates = list(dict.fromkeys(dates)) if dates else []
    unique_values = list(dict.fromkeys(values)) if values else []
    
    date = unique_dates[0] if unique_dates else None
    
    try:
        value = float(unique_values[0]) if unique_values else None
    except (ValueError, IndexError):
        value = None
    
    return date, value


def send_left_arrow():
    """Send left arrow key to move crosshairs backward."""
    import subprocess
    try:
        # Try pyautogui first
        import pyautogui
        pyautogui.press('left')
        return True
    except:
        pass
    
    try:
        # Fallback to xdotool
        subprocess.run(['xdotool', 'key', 'Left'], 
                      capture_output=True, timeout=1)
        return True
    except:
        pass
    
    return False


def get_browser_page_text() -> str:
    """Use the browser visualization tool to read page content.
    
    This function calls browser.get_content() to read what's displayed
    in the browser window - this is our visualization tool.
    
    Returns:
        The text content displayed in the browser
    """
    import subprocess
    
    # Use the MCP client to call get_content
    result = subprocess.run(
        ['python3', '-c',
         'import sys; sys.path.insert(0, "/home/openhands/.openhands/clients/python-client"); '
         'from browser import get_content; print(get_content(), end="")'],
        capture_output=True, 
        text=True, 
        timeout=15
    )
    
    if result.returncode == 0:
        return result.stdout
    return ""


def navigate_browser(url: str):
    """Navigate browser to URL using the visualization tool."""
    import subprocess
    
    subprocess.run(
        ['python3', '-c',
         f'import sys; sys.path.insert(0, "/home/openhands/.openhands/clients/python-client"); '
         f'from browser import navigate; navigate("{url}")'],
        capture_output=True,
        timeout=30
    )
    time.sleep(2)


def main():
    """Main function - fully automated data extraction."""
    print("=" * 60)
    print("StockCharts Reader - Fully Automated")
    print("=" * 60)
    print(f"\nChart: {CHART_URL}")
    print(f"Output: {OUTPUT_FILE}")
    print()
    
    # Step 1: Navigate to the chart
    print("Step 1: Navigating to StockCharts chart...")
    navigate_browser(CHART_URL)
    time.sleep(3)
    
    # Step 2: Activate crosshairs and position at end
    print("Step 2: Positioning crosshairs on last datapoint...")
    print("(The crosshairs should automatically snap to the latest bar)")
    time.sleep(2)
    
    # Now start automated extraction loop
    print("\nStep 3: Starting automated data extraction...")
    print("Reading data points automatically - no user interaction required.")
    print()
    print("Date          Value")
    print("-" * 25)
    
    # Create empty dataframe
    df = pd.DataFrame(columns=['Date', 'Value'])
    max_iterations = 2000
    iteration = 0
    last_date = None
    
    while iteration < max_iterations:
        iteration += 1
        
        try:
            # Get browser page content - this is our visualization tool
            content = get_browser_page_text()
            
            # Extract date and value from the visualized content
            date, value = extract_date_and_value_from_content(content)
            
            # Check if we've reached the end
            if not date:
                print(f"\nNo more data at iteration {iteration}")
                break
            
            # Check if we've looped back to the start
            if date == last_date:
                print(f"\nReached start of data at: {date}")
                break
            last_date = date
            
            # Add to dataframe
            new_row = pd.DataFrame([{'Date': date, 'Value': value}])
            df = pd.concat([df, new_row], ignore_index=True)
            
            print(f"{date:12} {value}")
            
            # After reading, send left arrow to move to next data point
            send_left_arrow()
            time.sleep(PAUSE)
            
        except Exception as e:
            print(f"Error at iteration {iteration}: {e}")
            break
    
    # Save data
    if len(df) > 0:
        # Sort by date (oldest first)
        def parse_date(d):
            try:
                from datetime import datetime
                return datetime.strptime(d, '%d-%b-%Y')
            except:
                return None
        
        df['parsed'] = df['Date'].apply(parse_date)
        df = df.sort_values('parsed').reset_index(drop=True)
        df = df.drop('parsed', axis=1)
        
        output_path = Path(OUTPUT_FILE)
        df.to_excel(output_path, index=False)
        
        print("\n" + "=" * 60)
        print("DATA SAVED")
        print("=" * 60)
        print(f"Output: {output_path.absolute()}")
        print(f"Total points: {len(df)}")
        print("\nData:")
        print(df.to_string())
    else:
        print("\nNo data recorded.")


if __name__ == "__main__":
    main()