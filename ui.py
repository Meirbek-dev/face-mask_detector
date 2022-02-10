import subprocess as sp
import sys
import tkinter as tk
import tkinter.font as tkFont

# BTN_IS_NOT_PRESSED = True
FILE_TO_RUN = ("python", "detect.py")


def main():
    window = tk.Tk()
    logo = tk.PhotoImage(file="logo.png")
    window.iconphoto(False, logo)
    # Установка заголовка окна
    window.title("Обнаружение маски")
    # Установка размера окна
    win_width = 290
    win_height = 95
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
    window['bg'] = "#ffffff"
    ft = tkFont.Font(family="Times Bold", size=14)

    def exit_cmd():
        sys.exit(0)

    def btn_detect_cmd():
        # global BTN_IS_NOT_PRESSED
        # if BTN_IS_NOT_PRESSED:
        sp.run(FILE_TO_RUN)
        #     btn_detect['text'] = 'Завершить обнаружение'
        #     btn_detect['fg'] = 'red'
        #     BTN_IS_NOT_PRESSED = False
        # elif not sp.run(FILE_TO_RUN):
        #     btn_detect["text"] = "Начать обнаружение"
        #     btn_detect['fg'] = 'black'
        #     BTN_IS_NOT_PRESSED = True
        # else:
        #     sp.Popen.terminate(sp.Popen(FILE_TO_RUN))  # closes the process
        #     BTN_IS_NOT_PRESSED = True

    btn_detect = tk.Button(window, bg='#ffffff', relief='solid', font=ft, command=btn_detect_cmd)
    btn_detect.pack(anchor='center', ipadx='3', padx='5', pady='5', side='top')
    btn_detect["text"] = "Начать обнаружение"  # Текст кнопки

    btn_exit = tk.Button(window, bg='#ffffff', relief='solid', font=ft, command=exit_cmd)
    btn_exit.pack(anchor='center', ipadx='3', padx='5', pady='5', side='top')
    btn_exit["text"] = "Выйти"

    window.mainloop()


if __name__ == "__main__":
    main()
    sys.exit()
