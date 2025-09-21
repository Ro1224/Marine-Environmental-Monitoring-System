from flask import Flask, render_template
import requests
from bs4 import BeautifulSoup

app = Flask(_name_)

@app.route('/cleanup-methods', methods=['GET'])
def cleanup_methods():
    # URL of the website containing cleanup methods
    url = 'https://www.marineinsight.com/environment/10-methods-for-oil-spill-cleanup-at-sea/'

    # Fetch the webpage content
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract cleanup methods and their respective content
    cleanup_methods = []
    
    # Find all headings (h3) and corresponding paragraphs (p)
    methods = soup.find_all('h3')  # Cleanup methods are usually under <h3> tags

    for method in methods:
        method_title = method.get_text(strip=True)
        method_description = method.find_next('p').get_text(strip=True)

        # Debugging output
        print(f"Title: {method_title}")
        print(f"Description: {method_description}\n")

        # Filter out unwanted entries like 'Explore' and 'More'
        if method_description and not method_title.startswith('Related Read') and not method_title.startswith('Leave a Reply') \
                and "Marine Engine" not in method_description and "Mooring" not in method_description \
                and 'Explore' not in method_title and 'More' not in method_title:
            cleanup_methods.append({
                'method': method_title,
                'description': method_description
            })

    print(f"Cleanup Methods: {cleanup_methods}")  # Debug: Check what data is being passed

    # Pass the cleanup methods to the HTML template
    return render_template('cleanup_strategies.html', cleanup_methods=cleanup_methods)


if _name_ == '_main_':
    app.run(debug=True)