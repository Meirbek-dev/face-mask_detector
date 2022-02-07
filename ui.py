import tkinter as tk
import tkinter.font as tkFont

import detect


def button_start_command():
    detect.main()


class App:
    def __init__(self, window):
        # Установка заголовка окна
        window.title("Обнаружение маски")
        # Установка размера окна
        win_width = 285
        win_height = 70
        # Получение разрешения экрана
        screenwidth, screenheight = window.winfo_screenwidth(), window.winfo_screenheight()
        # Установка окна по центру экрана
        alignstr = "%dx%d+%d+%d" % (
            win_width,
            win_height,
            (screenwidth - win_width) / 2,
            (screenheight - win_height) / 2,
        )
        window.geometry(alignstr)
        # Отключение расширения окна
        window.resizable(width=False, height=False)

        button_start = tk.Button(window)
        button_start.place(relx=.5, rely=.5, anchor="c")
        ft = tkFont.Font(family="Times Bold", size=14)
        button_start["font"] = ft
        button_start["text"] = "Начать обнаружение"  # Текст кнопки
        button_start["relief"] = "solid"  # Стиль границы кнопки
        button_start["command"] = button_start_command


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
