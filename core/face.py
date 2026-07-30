class Face():
    def __init__(self, color, stickers=None):
        self.color = color
        self.stickers = [color, color, color, color, color, color, color, color, color] if stickers is None else stickers
        

    def match(self, face):
        return all(face.stickers[x] == self.stickers[x] or face.stickers[x] == "" for x in range(9)) and face.color == self.color

    def __str__(self):
        return "Center: " + self.color + "\n" + str(self.stickers)

if __name__ == "__main__":
    f1 = Face("white")
    f2 = Face("white", ["white", "white", "white", "", "white", "white", "white", "white", ""])
    print(f1.match(f2))