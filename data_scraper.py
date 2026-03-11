import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import ssl
import urllib3

def scrape_team_stats(year):
    # Scrape regular season team stats
    
    url = f'https://sports-reference.com/cbb/seasons/{year}-school-stats.html'
    
    ssl._create_default_https_context = ssl._create_unverified_context
    # Make it look like it is coming from a browser
    try:
        tables = pd.read_html(url, header=[0,1])
        df = tables[0]
        df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
        
        print(f"Found {len(tables)} tables")
        
        print(f"Shape before cleaning: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:5]}...")
        
        first_col = df.columns[0]
        df = df[~df[first_col].astype(str).str.contains('Rk|Unnamed', na=False)]
        
        for col in df.columns:
            if 'Overall_G' in col or 'Overall_W' in col:
                df = df[~df[col].astype(str).str.contains('Overall', na=False)]
                break
        
        school_col = [col for col in df.columns if 'School' in col][0]
        
        # Create NCAA_Tournament column BEFORE removing NCAA from school names
        # Check if school name contains "NCAA"
        df['NCAA_Tournament'] = df[school_col].astype(str).str.contains('NCAA', na=False).astype(int)
        
        # Now remove "NCAA" from school names
        df[school_col] = df[school_col].astype(str).str.replace('NCAA', '', regex=False)
        df[school_col] = df[school_col].astype(str).str.replace(' NCAA', '', regex=False)
        df[school_col] = df[school_col].astype(str).str.replace('NCAA ', '', regex=False)
        df[school_col] = df[school_col].astype(str).str.strip()
        
        df = df.reset_index(drop=True)
        
        df.columns = df.columns.str.replace('School Advanced_', '', regex=False)
        df.columns = df.columns.str.replace(r'Unnamed: \d+_level_\d+_', '', regex=True)
        df.columns = df.columns.str.replace(r'Unnamed: \d+_level_\d+', '', regex=True)
        
        df = df.loc[:, df.columns.str.strip() != '']
        
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
        
        print(f"Shape after removing repeated headers: {df.shape}")
        
        return df
        
        
    except Exception as e:
        print(f"Error reading tables: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    time.sleep(5)
    
def scrape_advanced_team_stats(year):
    
    url = f'https://www.sports-reference.com/cbb/seasons/men/{year}-advanced-school-stats.html'

    ssl._create_default_https_context = ssl._create_unverified_context
    # Make it look like it is coming from a browser
    try:
        tables = pd.read_html(url, header=[0,1])
        df = tables[0]
        df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
        
        print(f"Found {len(tables)} tables")
        
        print(f"Shape before cleaning: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:5]}...")
        
        first_col = df.columns[0]
        df = df[~df[first_col].astype(str).str.contains('Rk|Unnamed', na=False)]
        
        for col in df.columns:
            if 'Overall_G' in col or 'Overall_W' in col:
                df = df[~df[col].astype(str).str.contains('Overall', na=False)]
                break
        
        school_col = [col for col in df.columns if 'School' in col][0]
        
        # Create NCAA_Tournament column BEFORE removing NCAA from school names
        # Check if school name contains "NCAA"
        df['NCAA_Tournament'] = df[school_col].astype(str).str.contains('NCAA', na=False).astype(int)
        
        # Now remove "NCAA" from school names
        df[school_col] = df[school_col].astype(str).str.replace('NCAA', '', regex=False)
        df[school_col] = df[school_col].astype(str).str.replace(' NCAA', '', regex=False)
        df[school_col] = df[school_col].astype(str).str.replace('NCAA ', '', regex=False)
        df[school_col] = df[school_col].astype(str).str.strip()
        
        df = df.reset_index(drop=True)
        
        df.columns = df.columns.str.replace('School Advanced_', '', regex=False)
        df.columns = df.columns.str.replace(r'Unnamed: \d+_level_\d+_', '', regex=True)
        df.columns = df.columns.str.replace(r'Unnamed: \d+_level_\d+', '', regex=True)
        
        df = df.loc[:, df.columns.str.strip() != '']
        
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
            
        print(f"Shape after removing repeated headers: {df.shape}")
        
        return df
        
        
    except Exception as e:
        print(f"Error reading tables: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    time.sleep(5)

# Combine both pages of stats for easier readablity
def combine_stats(basic_df, advanced_df, year):
    basic_school_col = [col for col in basic_df.columns if 'School' in col][0]
    advanced_school_col = [col for col in advanced_df.columns if 'School' in col][0]
    
    combined = pd.merge(basic_df, advanced_df, left_on=basic_school_col, right_on=advanced_school_col, how='inner', suffixes=('', '_drop'))
    
    combined = combined[[col for col in combined.columns if not col.endswith('_drop')]]
    
    combined['Year'] = year
    
    return combined
    
def main():
    
    if not os.path.exists('data'):
        os.makedirs('data')
        
    years = range(2000, 2005)
    
    for year in years:
        print(f"Scraping {year}...")
        try:
            basic = scrape_team_stats(year)
            print(f"Scraped {year} stats")
            time.sleep(5)
            advanced = scrape_advanced_team_stats(year)
            print(f"Scraped {year} advanced stats.")
            combined = combine_stats(basic, advanced, year)
            print(f"Combined: {combined.shape}")
            combined.to_csv(f'data/team_stats_{year}.csv', index=False)
            print(f"Saved {year} successfully.")
        except Exception as e:
            print(f"Error with {year}: {e}")
        
        time.sleep(5)
        
if __name__ == "__main__":
    main()