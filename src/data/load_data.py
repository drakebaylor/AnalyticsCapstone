import requests
import time

def load_data():
    #MLB.com offers the easiest method for obtaining all the active players in the 2025 season.
    url = "https://statsapi.mlb.com/api/v1/sports/1/players?fields=people,fullName,lastName,nameSlug&season=2025"

    response = requests.get(url)
    response_json = response.json()
    
    #From this json, we need to extract the players' names and add it to a list of active players
    active_players = []
    for player in response_json['people']:
        active_players.append(player['fullName'])
    
    #Baseball-Reference contains a wealth of stats on each player's home page. For now, we will request the entire HTML of each player's page.
    #Baseball-Reference urls follow a particular pattern. In general for each player, I need to parse the first two letters of their first name and first five of their last name.
    #We will loop over the active_players list with that in mind, add error players to a list for additional processing.
    for player in active_players:
        first_name = player.split(" ")[0]
        last_name = player.split(" ")[1]
        first_name_parsed = first_name[0:2].lower()
        last_name_parsed = last_name[0:5].lower()
    
        time.sleep(2.1) #<---- Slows down the web call to prevent rate limiting. 
        url = f"https://www.baseball-reference.com/players/{last_name_parsed[0]}/{last_name_parsed}{first_name_parsed}01.shtml"
        response = requests.get(url)
        if "Page Not Found (404 error)" not in response.text:
            with open(f'../data/raw/{player}.html', 'w+', encoding='utf-8') as f:
                f.write(response.text)
                f.close()
    
            print(f"{player} html file downloaded.")
        else:
            print(f"{player} html file failed to download. Added to error players.")
            

if __name__ == "__main__":
    load_data()
    print("All active players' html files downloaded.")
    print("Please check the data/raw folder for the files.")
    print("If any files failed to download, please check the console output for the list of error players.")