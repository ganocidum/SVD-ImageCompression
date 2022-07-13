import numpy as np
import numpy.linalg as la
import cv2
from tkinter import StringVar, Tk, Button, Label, Frame, messagebox, filedialog
from tkinter.constants import CENTER, RIGHT, LEFT, BOTTOM
from tkinter.ttk import * 

image_path = ''
U, sigma, V_T = None, None, None
max_rank = 0
rank = -1

#                                               SVD
#=====================================================================================================
def calculate_eigh_ATA(A):
    '''
        Calculate the eigenvalues and eigenvectors of matrix A^T.A 
        Arguments:
            A: numpy array - the image
        Returns:
            eigenvalues: numpy array
            eigenvectors: numpy array
    '''
    AT_A = np.dot(A.T, A)
    eigenvalues, eigenvectors = la.eigh(AT_A)
    eigenvalues = np.maximum(eigenvalues, 0.)

    sorted_index = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sorted_index]
    eigenvectors = eigenvectors[:, sorted_index]

    return eigenvalues, eigenvectors

def calculate_svd(A):
    '''
        Using SVD to calculate U, sigma and V^T matrices of matrix A
        Arguments:
            A: numpy array - the image
        Returns:
            U: numpy array
            sigma: numpy array
            V_T: numpy array
    '''
    m = A.shape[0]
    n = A.shape[1]
    eigenvalues, eigenvectors = calculate_eigh_ATA(A)
    
    sigma = np.zeros([m, n])
    for i in range(min(m, n)):
        sigma[i][i] = max(eigenvalues[i], 0.)
    sigma = np.maximum(np.sqrt(sigma), 0)

    V = eigenvectors
    V_T = V.T
    
    U = np.zeros([m, m])
    for i in range(m):
        U[:, i] = 1/sigma[i][i] * np.dot(A, V[:, i])

    return U, sigma, V_T 


def find_A_approx(A, rank):
    '''
        Calculate the matrix A approximately of A with rank using SVD
        Arguments:
            A: numpy array - the image
            rank: int - the rank of the approximate matrix, 
                the greater the rank is the more accuracy the approximate image is
        Returns:
            result: numpy array - the approximately image
            error: double - the error of the approximate image
    '''
    U, sigma, V_T = calculate_svd(A)
    new_A = np.zeros(A.shape)
    new_A = U[:, :rank] @ sigma[:rank, :rank] @ V_T[:rank, :]
    if rank < min(A.shape[0], A.shape[1]):
      error = np.sum(sigma[rank:, :])/ np.sum(sigma)
    else: 
      error = 0.
    return new_A, error

#=====================================================================================================
def makeCenter(root):
    '''
    Place the window of the app in the center of the screen
        Arguments:
            root: a tkinter instance
        
        Returns:
            None
    '''
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth()//2) - (width//2)
    y = (root.winfo_screenheight()//2) - (height//2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))

def get_input_file():
    '''
        Get image path from user
        Arguments:
            None
        Returns:
            None
    '''
    global image_path
    image_path = filedialog.askopenfilename(initialdir="\\", title = "Select a file", filetypes = (("all files", "*.*"), ("image files", "*.jpg"), ("image files", "*.png"), ("image files", "*.jpeg")))
    if len(image_path)!=0: 
        input_label.config(text=image_path, foreground='green')
    else:
        input_label.config(text="Please choose input file", foreground='red')

#                                         GRAY IMAGE
#=====================================================================================================
def show_gray_image():
    '''
        Get rank from user to calculate and show approximate image
        Arguments:
            None
            
        Returns:
            None
    '''
    global A_approx, rank
    try:
        rank = rank_value.get()
        rank = int(rank)
        A_approx, error = find_A_approx(gray_image, rank)
        error_label.config(text='Error of approximation: ' + str(error))
        cv2.imshow('Approximate image', A_approx)
        cv2.waitKey(0)
    except:
        messagebox.showerror('Error', 'Rank must be a positive integer <= '+ str(max_rank) +' !!!')

def save_gray_image():
    '''
        Save the re-constructed image
        Arguments:
            None
        
        Returns:
            None
    '''
    global A_approx, rank
    if len(image_path)>0 and rank>=0:
        try:
            A_approx *= 255
            output_path = filedialog.asksaveasfilename(initialdir="\\", title = "Select a file", filetypes = (("all files", "*.*"), ("image files", "*.jpg"), ("image files", "*.png"), ("image files", "*.jpeg")))
            if len(output_path)!=0:
                if not (output_path.endswith('.jpg')):
                    output_path += '.jpg'
                messagebox.showinfo('Info', 'Save successfully !!!')
                cv2.imwrite(output_path, A_approx)
        except Exception as e:
            print(e)
            messagebox.showerror('Error', "Please try again !!!")
    else:
        messagebox.showerror('Error', "Please choose image or enter rank !!!")

def process_gray_image():
    '''
        Process the image to calculate U, sigma, V_T and max_rank
        Arguments:
            None
        Returns:
            None
    '''
    global gray_image, max_rank
    try:
        default_image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(default_image, cv2.COLOR_BGR2GRAY)
        gray_image = gray_image/255.
        gray_image = np.array(gray_image)
        U, sigma, V_T = calculate_svd(gray_image)
        max_rank = la.matrix_rank(sigma)
        process_label.config(text=str(max_rank))
    except Exception as e:
        print(e)
        messagebox.showerror('Error', 'This is not an image, please try again !!!')

#                                           COLOR IMAGE
#=====================================================================================================
def show_color_image():
    '''
        Get rank from user to calculate and show approximate image
        Arguments:
            None
            
        Returns:
            None
    '''
    global new_color_image, rank
    try:
        rank = rank_value.get()
        rank = int(rank)
        red_compressed, er = find_A_approx(red_channel, rank)
        green_compressed, er = find_A_approx(green_channel, rank)
        blue_compressed,er = find_A_approx(blue_channel, rank)
        new_color_image = np.stack((red_compressed, green_compressed, blue_compressed), axis=2)
        cv2.imshow('Approximate image', new_color_image)
        cv2.waitKey(0)
    except:
        messagebox.showerror('Error', 'Rank must be a positive integer <= '+ str(max_rank) +' !!!')

def save_color_image():
    '''
        Save the re-constructed image
        Arguments:
            None
        
        Returns:
            None
    '''
    global new_color_image, rank
    if len(image_path)>0 and rank>=0:
        try:
            new_color_image *= 255
            output_path = filedialog.asksaveasfilename(initialdir="\\", title = "Select a file", filetypes = (("all files", "*.*"), ("image files", "*.jpg"), ("image files", "*.png"), ("image files", "*.jpeg")))
            if len(output_path)!=0:
                if not (output_path.endswith('.jpg')):
                    output_path += '.jpg'
                messagebox.showinfo('Info', 'Save successfully !!!')
                cv2.imwrite(output_path, new_color_image)
        except Exception as e:
            print(e)
            messagebox.showerror('Error', "Please try again !!!")
    else:
        messagebox.showerror('Error', "Please choose image or enter rank !!!")

def process_color_image():
    '''
        Process the image to calculate U, sigma, V_T and max_rank
        Arguments:
            None
        Returns:
            None
    '''
    global red_channel, green_channel, blue_channel, max_rank
    try:
        default_image = cv2.imread(image_path)
        red_channel = default_image[:, :, 0]/255.
        green_channel = default_image[:, :, 1]/255.
        blue_channel = default_image[:, :, 2]/255.
        max_rank = la.matrix_rank(blue_channel)
        process_label.config(text=str(max_rank))
    except Exception as e:
        print(e)
        messagebox.showerror('Error', 'This is not an image, please try again !!!')

#                                             MAIN
#=====================================================================================================
root = Tk()
root.title('SVD')
root.geometry("600x300")
makeCenter(root)
root.resizable(width=False, height=False)
label = Label(root, text='Image Reconstruction', font=(20)).pack(pady=5)

#--------------------------------------------------------------------
input_frame = Frame(root)
input_button = Button(input_frame, width=20, text = 'Choose image', command=get_input_file)
input_label = Label(input_frame, width=60)

process_frame = Frame(root)
process_gray_button = Button(process_frame, width=20, text = 'Process image', command=process_gray_image)
process_label = Label(process_frame, width=60, foreground='green')

process_color_button = Button(process_frame, width=20, text = 'Process image', command=process_color_image)



rank_frame = Frame(root)
rank_value = StringVar()
rank_label = Label(rank_frame, width=20, text='Rank', anchor=CENTER)
rank_entry = Entry(rank_frame, width=40, textvariable=rank_value)
rank_gray_button = Button(rank_frame, width=20, text='OK', command=show_gray_image)

rank_color_button = Button(rank_frame, width=20, text='OK', command=show_color_image)



error_label = Label(root)

save_gray_button = Button(root, text='Save image', command=save_gray_image)
save_color_button = Button(root, text='Save image', command=save_color_image)

#--------------------------------------------------------------------
def close_page1():
    if input_button.winfo_exists():
        input_button.pack_forget()
    if input_label.winfo_exists():
        input_label.pack_forget()
        input_label.config(text='')
    if input_frame.winfo_exists:
        input_frame.pack_forget()

    if process_gray_button.winfo_exists:
        process_gray_button.pack_forget()
    if process_label.winfo_exists():
        process_label.pack_forget()
        process_label.config(text='')
    if process_frame.winfo_exists():
        process_frame.pack_forget()

    if rank_label.winfo_exists():
        rank_label.pack_forget()
        rank_label.config(text='')
    if rank_gray_button.winfo_exists():
        rank_gray_button.pack_forget()
    if rank_entry.winfo_exists():
        rank_entry.pack_forget()
    if rank_frame.winfo_exists():
        rank_frame.pack_forget()

    if error_label.winfo_exists():
        error_label.pack_forget()
        error_label.config(text='')

    if save_gray_button.winfo_exists():
        save_gray_button.pack_forget()
def close_page2():
    if input_button.winfo_exists():
        input_button.pack_forget()
    if input_label.winfo_exists():
        input_label.pack_forget()
        input_label.config(text='')
    if input_frame.winfo_exists:
        input_frame.pack_forget()

    if process_color_button.winfo_exists:
        process_color_button.pack_forget()
    if process_label.winfo_exists():
        process_label.pack_forget()
        process_label.config(text='')
    if process_frame.winfo_exists():
        process_frame.pack_forget()

    if rank_label.winfo_exists():
        rank_label.pack_forget()
        rank_label.config(text='')
    if rank_color_button.winfo_exists():
        rank_color_button.pack_forget()
    if rank_entry.winfo_exists():
        rank_entry.pack_forget()
    if rank_frame.winfo_exists():
        rank_frame.pack_forget()

    if save_color_button.winfo_exists():
        save_color_button.pack_forget()
def page1():
    close_page2()
    input_button.pack(side=LEFT)
    input_label.pack()
    input_frame.pack(pady=10)

    process_gray_button.pack(side=LEFT)
    process_label.pack()
    process_frame.pack(pady=10)

    rank_label.pack(side=LEFT)
    rank_gray_button.pack(side=RIGHT)
    rank_entry.pack()
    rank_frame.pack(pady=10)

    error_label.pack()

    save_gray_button.pack()

def page2():
    close_page1()
    input_button.pack(side=LEFT)
    input_label.pack()
    input_frame.pack(pady=10)

    process_color_button.pack(side=LEFT)
    process_label.pack()
    process_frame.pack(pady=10)

    rank_label.pack(side=LEFT)
    rank_color_button.pack(side=RIGHT)
    rank_entry.pack()
    rank_frame.pack(pady=10)

    save_color_button.pack()

page1_button = Button(root, text="Gray Image", command=page1)
page2_button = Button(root, text="Color Image", command=page2)

page1_button.pack()
page2_button.pack()

root.mainloop()