from bs4 import BeautifulSoup
import requests
import json

# Text file containing relevant html for teacher cards
# TODO : Find a better way to scrape this data
with open("profs.txt", "r") as filein:
    html = filein.read()

# Read from the html and get all professor numbers
soup = BeautifulSoup(html, "html.parser")
teacher_links = [
    a["href"]
    for a in soup.find_all(
        "a", class_="TeacherCard__StyledTeacherCard-syjs0d-0 dLJIlx", href=True
    )
]


# Iterate through professors getting comment and rating for each review left
data = []  # list of dictionaries with rating : comment entries
for link in teacher_links:
    url = "https://www.ratemyprofessors.com/" + link
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")
    prof_name = soup.find(
        "span", class_="NameTitle__LastNameWrapper-dowf0z-2 glXOHH"
    ).text
    comments = [
        comment.text
        for comment in soup.find_all(
            "div", class_="Comments__StyledComments-dzzyvm-0 gRjWel"
        )
    ]
    ratings = [
        int(float(rating.text)) - 1
        for rating in soup.find_all(
            "div",
            class_=lambda x: x
            and x.startswith("CardNumRating__CardNumRatingNumber-sc-17t4b9u-2 "),
        )
    ]
    ratings = ratings[::2]

    if len(ratings) != len(comments):
        continue

    for i in range(len(ratings)):
        review = {"comment": comments[i], "rating": ratings[i]}
        data.append(review)


# Dump all the ratings
with open("reviews.json", "w") as json_file:
    json.dump(data, json_file, indent=4)
