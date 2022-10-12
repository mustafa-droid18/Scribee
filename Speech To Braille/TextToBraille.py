import os
from PIL import Image, ImageDraw, ImageOps

Dictionary = {
    " ": "000000",
    "a": "100000",
    "b": "110000",
    "c": "100100",
    "d": "100110",
    "e": "100010",
    "f": "110100",
    "g": "110110",
    "h": "110010",
    "i": "010100",
    "j": "010110",
    "k": "101000",
    "l": "111000",
    "m": "101100",
    "n": "101110",
    "o": "101010",
    "p": "111100",
    "q": "111110",
    "r": "111010",
    "s": "011100",
    "t": "011110",
    "u": "101001",
    "v": "111001",
    "w": "010111",
    "x": "101101",
    "y": "101111",
    "z": "101011",
    "0": "010110",
    "1": "100000",
    "2": "110000",
    "3": "100100",
    "4": "100110",
    "5": "100010",
    "6": "110100",
    "7": "110110",
    "8": "110010",
    "9": "010100",
    "-": "001001",
    ":": "010010",
    ".": "010011",
    ",": "010000",
    "'": "001000",
    "!": "011010",
    "?": "011001",
    ";": "011000",
    "@": "000000",
    "NUMERIC": "001111",
    "CAPITAL": "000001"
    #"#": "001111",
    #"+": "001101",
    #"*": "100001",
    #"=": "111111",
    #"<": "110001",
    #">": "001110",
    #"(": "111011",
    #")": "011111",
}

FONT_SIZE = 5
FONT_COLOR = "black"
MARGIN = 75
X_SPACING, Y_SPACING = 6, 20
PAPER_COLOR = "white"
PAPER_WIDTH = 850
PAPER_HEIGHT = 1100

class Character:
    def __init__(self, b_code, pixel_size=FONT_SIZE):

        b_code = str(b_code)  # Turn into a list
        self._list = "000000000000000"  # List that represents which pixels to fill, is a 3 * 5 rectangle

        self.pixel_size = pixel_size
        self.width = 3
        self.height = 5

        self.pixels = [[0 for n in range(3)].copy() for i in range(5)]  # List to be used by Paper to display a char

        # Update display_list to match the braille code, see http://braillebug.org/braille_deciphering.asp
        self.pixels[0][0] = int(float(b_code[0]))
        self.pixels[2][0] = int(float(b_code[1]))
        self.pixels[4][0] = int(float(b_code[2]))
        self.pixels[0][2] = int(float(b_code[3]))
        self.pixels[2][2] = int(float(b_code[4]))
        self.pixels[4][2] = int(float(b_code[5]))

class Paper:
    def __init__(self, name, page=1, charset=False, width=PAPER_WIDTH, height=PAPER_HEIGHT, color=PAPER_COLOR):
        """Creates an image that is associated with the object"""

        self.name = str(name)  # Name to be used when saving the file
        self._height = height
        self._width = width
        self._color = color
        self._page = page

        if charset:
            self.charset = charset
        else:
            self.charset = Dictionary  # Converts the dictionary into one with objects
            for x in self.charset:
                self.charset[x] = Character(self.charset[x])

        self._clear()  # Sets the image to a blank page

    def _clear(self):
        """Resets the image to its default color"""

        self.image = Image.new("RGB", (self._width, self._height), self._color)

    def draw(self, x, y, dx, dy, color):
        """Draws a colored rectangle onto the image using the coordinates of the top left and its size"""

        draw = ImageDraw.Draw(self.image)

        draw.rectangle([(x,y),(dx,dy)], color, outline=None)

    def save(self):
        """Saves the image to a physical file that is the name the object was created with"""

        self.image.save("./output/" + self.name + " pg" + str(self._page) + ".png")

    def show(self):
        """Opens the image in whatever is your system default, doesn't require saving"""

        self.image.show()

    def drawChar(self, char, x, y, color=FONT_COLOR):
        """Takes a Character object and draws it on the image at the given coordinates using parameters inside the
        Character"""

        pixels, width, height = char.pixels, char.width, char.height
        pixel_size = char.pixel_size
        dx, dy = 0, 0

        # Loops though the character's list that specifies where to draw
        for row in range(char.height):

            for column in range(char.width):

                if pixels[row][column]:  # If there is a 1 at the specified index in the char, draw a pixel(s)
                    self.draw(x + dx, y + dy, x + dx + pixel_size, y + dy + pixel_size, color)

                dx += pixel_size + 1  # Increase the horizontal offset

            dy += pixel_size + 1  # Increase the vertical offset
            dx = 0  # Reset the horizontal offset

    def convertBrailleCharacter(self, string):
        # First convert the string into braille letters
        braille_code = []

        numeric_conditions = False
        for letter in string:

            if letter.isupper():  # Checks for special cases
                braille_code.append(self.charset["CAPITAL"])
                braille_code.append(self.charset[letter.lower()])
                numeric_conditions = False

            elif letter.isnumeric():
                if not numeric_conditions:
                    braille_code.append(self.charset["NUMERIC"])
                    numeric_conditions = True

                braille_code.append(self.charset[letter])

            else:  # Normal condition
                braille_code.append(self.charset[letter])
                numeric_conditions = False

        return braille_code

    def drawSentence(self, braille_code, x=MARGIN, y=MARGIN,
                     wrap_width=(PAPER_WIDTH - (MARGIN * 2)),
                     x_spacing=X_SPACING,
                     y_spacing=Y_SPACING,
                     color=FONT_COLOR):
        """Draws a sentence starting at a point, wraps after passing a specified width
        (relative to the left edge of paper), requires a list of braille character objects, will create multiple pages"""

        dx, dy = 0, 0
        character_width = FONT_SIZE * 3 + x_spacing * 2
        character_height = FONT_SIZE * 5 + y_spacing

        # Displaying the letters
        for n in range(len(braille_code)):

            character = braille_code[n]
            self.drawChar(character, x + dx, y + dy, color)

            if dx + character_width >= wrap_width:  # If it has hit the right margin, wrap
                dx = 0
                dy += character_height
            else:
                dx += character_width  # Move to next char

            if dy + character_height >= PAPER_HEIGHT - MARGIN * 2:  # If it hits the end of the page
                # Make a new Paper object, have it draw remaining chars
                next_page = Paper(self.name, (self._page + 1), self.charset)
                next_page.drawSentence(braille_code[n:], x, y, wrap_width, x_spacing, y_spacing, color)
                break

        self.save()

def getInput(file):  # To open the file
    input_text = open(file, "r")
    return input_text.read().replace('\n', '')


def main():

    exit = False

    while not exit:

        #  Main Menu

        print("Welcome to the Scribee!: Your Communication Companion"
              "\n====="
              "\n(1) Convert input from console"
              "\n(2) Convert input from text file"
              "\n(3) Exit program")

        input_valid = False
        while not input_valid:
            choice = input("Enter option:")
            if choice in ["1", "2", "3", "4"]:
                choice = int(choice)
                input_valid = True
            else:
                print("Invalid input, try again.")
                

        if choice == 1:  # If user chose console input
                print("=" * 25)
                input_valid = False
                while not input_valid:
                    text = input("Enter text to convert:")
                    for letter in text:
                        if Dictionary.DICTIONARY.get(letter) is None and not letter.isupper():  # Make sure console input is valid
                            print("'" + letter + "' is an invalid character, try again.")
                            input_valid = False
                            break
                        else:
                            input_valid = True
        
        if choice == 2:  # If user chose to read from file
            print("=" * 25)
            input_valid = False
            while not input_valid:
                name = input("Enter name of paper:")
                if len(name) > 0:
                    input_valid = True
                else:
                    print("Invalid input, try again.")

                paper_name = name
            text = getInput("file_title.txt")
            for letter in text:
                if Dictionary.get(letter) is None and not letter.isupper():  # Make sure text is valid
                    print("'" + letter + "' is an invalid character."
                                         "\nFix text and run the program again.")
                    input("Press Enter to exit...")
                    quit()  # If input.txt has invalid char, exit program

        if not os.path.isdir("./output"):  # If output folder doesn't exist, make one
                os.mkdir("./output")

        paper = Paper(paper_name)

        print("=" * 25 + "\nWorking...")
        
        paper.drawSentence(paper.convertBrailleCharacter(text))
        print("Done.")

         # Exit

        if choice == 3:
            print("Exiting...")
            exit = True

    print("\n\n\n\n\n")


if __name__ == "__main__":
    main()
