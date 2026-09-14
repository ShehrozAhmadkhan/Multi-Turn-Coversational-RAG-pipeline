import PyPDF2

def load_data(file_path):
    file = open(file_path,"rb")
    book = PyPDF2.PdfReader(file)
    pdfinstring = ""

    for page in book.pages:
        text = page.extract_text()
        if text:
            pdfinstring += text + "\n"

    file.close()

    return pdfinstring

"""
data = load_data("catalog.pdf")


search_text = "COMP 200"

positions = []
start = 0

while True:
    position = data.find(search_text, start)

    if position == -1:
        break

    positions.append(position)
    start = position + 1

print("Total occurrences:", len(positions))

for position in positions:
    print("\n--- MATCH ---")
    print(data[position:position + 400])

"""