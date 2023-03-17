import cv2

# Cargar la imagen
img = cv2.imread("grupo.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Cargar el clasificador para deteccion, requiere cv2.data para obtener la ruta completa
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

faces = face_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Dibujar un rectangulo en las caras identificadas
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Print the number of detected faces
cv2.putText(img, "Cantidad de rostros detectados: {}".format(
    len(faces)), (0, 285), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
cv2.imshow("Output", img)
cv2.waitKey(0)
