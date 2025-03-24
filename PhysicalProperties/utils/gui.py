import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from propfunc.figures import figures

initial_values = {
    "To": 900,
    "Ti": 300,
    "Pout": 3.45e6,
    "component": 19,
    "density_method": "RK_PR",
}

def loop():
    def get_input():
        # 获取输入框中的值
        To = entry_To.get()
        Ti = entry_Ti.get()
        Pout = entry_Pout.get()
        component = entry_component.get()
        density_method = entry_density_method.get()

        # 显示输入结果
        messagebox.showinfo("输入", f"出口温度: {To}\n入口温度: {Ti}\n出口压力: {Pout}\n组分数: {component}\nmethod: {density_method}")
        
    def show_plot():
        # 获取输入框中的值
        To = entry_To.get()
        Ti = entry_Ti.get()
        Pout = entry_Pout.get()
        component = entry_component.get()
        density_method = entry_density_method.get()
        # 将字符串转换为数值类型
        try:
            To = float(To)  # 转换为浮点数
            Ti = float(Ti)  # 转换为浮点数
            Pout = float(Pout)  # 转换为浮点数
            component = int(component)  # 转换为整数
        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的数字！")

        # 清空之前的图表（如果有）
        for widget in plot_frame.winfo_children():
            widget.destroy()

        # 调用绘图函数
        fig = figures(To, Ti, Pout, 60, component, np.array([1] + [0] * (component - 1)), density_method, export_to_excel=False, export_path='RK_PR.xlsx')

        # 将图表嵌入到 tkinter 窗口中
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # 创建主窗口
    root = tk.Tk()
    root.title("物性计算方法")

    # 创建标签和输入框
    label_To = tk.Label(root, text="温度终值:")
    label_To.grid(row=0, column=0, padx=5, pady=5)
    entry_To = tk.Entry(root)
    entry_To.grid(row=0, column=1, padx=5, pady=5)
    entry_To.insert(0, initial_values["To"])  # 设置初始值

    label_Ti = tk.Label(root, text="温度初值:")
    label_Ti.grid(row=1, column=0, padx=5, pady=5)
    entry_Ti = tk.Entry(root)
    entry_Ti.grid(row=1, column=1, padx=5, pady=5)
    entry_Ti.insert(0, initial_values["Ti"])  # 设置初始值

    label_Pout = tk.Label(root, text="出口压力:")
    label_Pout.grid(row=2, column=0, padx=10, pady=10)
    entry_Pout = tk.Entry(root)
    entry_Pout.grid(row=2, column=1, padx=10, pady=10)
    entry_Pout.insert(0, initial_values["Pout"])  # 设置初始值

    label_component = tk.Label(root, text="组分数:")
    label_component.grid(row=3, column=0, padx=10, pady=10)
    entry_component = tk.Entry(root)
    entry_component.grid(row=3, column=1, padx=10, pady=10)
    entry_component.insert(0, initial_values["component"])  # 设置初始值

    label_density_method = tk.Label(root, text="method:")
    label_density_method.grid(row=4, column=0, padx=10, pady=10)
    entry_density_method = tk.Entry(root)
    entry_density_method.grid(row=4, column=1, padx=10, pady=10)
    entry_density_method.insert(0, initial_values["density_method"])  # 设置初始值

    # 创建提交按钮
    submit_button = tk.Button(root, text="确认", command=show_plot)
    submit_button.grid(row=5, column=0, columnspan=2, pady=10)

    # 创建一个框架用于放置图表
    plot_frame = tk.Frame(root)
    plot_frame.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

    # 运行主循环
    root.mainloop()
    
if __name__ == '__main__':
    loop()