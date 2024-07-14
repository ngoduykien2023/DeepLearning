# Tensorflow: là thư viện mã nguồn mở được phát triển bởi Google, được sử dụng cho việc tính toán số học sử dụng đồ thị luồng dữ liệu.
# Numpy: là thư viện hỗ trợ cho việc tính toán các mảng nhiều chiều. Numpy cực kì hữu ích khi thực hiện các hàm liên quan đến Đại số tuyến tính.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Trước hết, chúng ta sử dụng keras.datasets để load dữ liệu để hiển thị một số thông tin về dataset.
# Tải dữ liệu thời trang
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# In hình dạng tập huấn luyện
# Lưu ý: có 60.000 dữ liệu huấn luyện có kích thước hình ảnh 28x28, 60.000 nhãn tập huấn.
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

# In số lượng tập dữ liệu đào tạo và kiểm tra
print('Train set:', x_train.shape[0])
print('Test set:', x_test.shape[0])

# Xác định các nhãn
fashion_mnist_labels = [
  "T-shirt/top",  # index 0
  "Trouser",      # index 1
  "Pullover",     # index 2 
  "Dress",        # index 3 
  "Coat",         # index 4
  "Sandal",       # index 5
  "Shirt",        # index 6 
  "Sneaker",      # index 7 
  "Bag",          # index 8 
  "Ankle boot"    # index 9
]

# Chọn bất kỳ 1 số trong khoảng từ 0 đến 59,999
img_index = 5

# y_train chứa các nhãn, từ 0 đến 9 (index 0 -> 9 tương ứng với mỗi loại fashion)
label_index = y_train[img_index]

# In nhãn, ví dụ nhãn thứ 2 là Pullover (áo phông)
print("y = " + str(label_index) + " " +(fashion_mnist_labels[label_index]))

# Hiển thị một trong những hình ảnh từ tập dữ liệu đào tạo
plt.imshow(x_train[img_index])

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
print("Số lượng dữ liệu train: " + str(len(x_train)))
print("Số lượng dữ liệu test: " + str(len(x_test)))

# Chia nhỏ dữ liệu thành các tập huấn luyện
# Trích từ tập training data ra một tập con nhỏ và thực hiện việc đánh giá mô hình trên tập con nhỏ này. Tập con nhỏ được trích ra từ training set này được gọi là validation set
# Đặt 5000 vào validation set và giữ 55.000 còn lại cho training set
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# Định hình lại dữ liệu đầu vào từ (28, 28) thành (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
# One-hot encoding là một quá trình mà các biến phân loại (label) được chuyển đổi thành một mẫu có thể cung cấp cho các thuật toán để thực hiện công việc tốt hơn khi mà dự đoán.
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# In tập huấn luyện
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

# In số lượng tập dữ liệu of training, validation, and test datasets
print('Train set:', x_train.shape[0])
print('Validation set:', x_valid.shape[0],)
print('Test set:', x_test.shape[0],)

# Khởi tạo models Sequential()
model = tf.keras.Sequential()

# Xác định hình dạng đầu vào trong lớp đầu tiên của neural network
# Tạo Convolutionnal Layers: Conv2D là convolution dùng để lấy feature từ ảnh với các tham số
# filters: số filter của convolution
# kernel_size: kích thước window search trên ảnh
# activation: chọn activation như linear, softmax, relu, tanh, sigmoid (relu là hàm trả về giá trị tích cực, nhưng không trả lại giá trị âm)
# padding: có thể là "valid" hoặc "same". Với same thì có nghĩa là padding =1.
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)))
# Hàm MaxPooling2D hoặc AvergaPooling1D, 2D (lấy max , trung bình) với từng size.
# pool_size: kích thước ma trận để lấy max hay average
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
# Dropout: chống over-fitting
# Hiểu đơn giản là, trong mạng neural network, kỹ thuật dropout là việc chúng ta sẽ bỏ qua một vài unit trong suốt quá trình train trong mô hình, những unit bị bỏ qua được lựa chọn ngẫu nhiên.
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

# Flatten dùng để lát phằng layer để fully connection (ví dụ: shape: 28x28 qua layer này sẽ là 784x1)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Bảng tóm tắt mô hình
model.summary()
# Compile mô hình
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(
    filepath='model.weights.best.keras',
    verbose = 1,
    save_best_only = True
)

# Batch_size: số lượng mẫu mà Mini-batch GD sử dụng cho mỗi lần cập nhật trọng số.
# Ta có tập huấn luyện gồm 55.000 hình ảnh, chọn batch-size là 100 images có nghĩa là mỗi lần cập nhật trọng số, ta dùng 100 images. Lúc đó ta mất 55.000/100 = 550 iterations (số lần lặp) để duyệt qua hết tập huấn luyện (hoàn thành 1 epochs). Có nghĩa là khi dữ liệu quá lớn, chúng ta không thể đưa cả tập data vào train được, ta phải chia nhỏ data ra thành nhiều batch nhỏ hơn.
# batch_size (default = 1): Batch size ảnh hưởng đến chất lượng học của mô hình, batch size càng lớn thì mô hình sẽ càng học được tốt hơn.
# Epoch là số lần duyệt qua hết số lượng mẫu trong tập huấn luyện.
model.fit(
    x_train,
    y_train,
    batch_size = 100,
    epochs = 2,
    validation_data=(x_valid, y_valid),
    callbacks=[checkpointer]
)

# Kiểm tra mô hình bằng cách sử dụng model.predict dữ liệu thử nghiệm
y_check = model.predict(x_test)

# Lấy ngẫu nhiên 15 hình ảnh từ tập test để dự đoán
figure = plt.figure(figsize=(20,10)) # Đặt kích thức hình ảnh
for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)): # ramdom 15 ảnh từ tập dữ liệu test
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    
    # Hiển thị từng hình ảnh
    ax.imshow(np.squeeze(x_test[index]))
    predict_index = np.argmax(y_check[index])
    true_index = np.argmax(y_test[index])

    # Đặt tiêu đề cho mỗi hình ảnh
    ax.set_title("{} ({})".format(
        fashion_mnist_labels[predict_index],
        fashion_mnist_labels[true_index]),
        color=("green" if predict_index == true_index else "red")
    )

    # Đánh giá mô hình trên test set
score = model.evaluate(x_test, y_test, verbose=0)

# Kiểm tra độ chính xác
print("Test loss:", score[0])
print("Test accuracy:", score[1])
plt.savefig("file.png")



