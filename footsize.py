import cv2
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
import random as rng

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    img = img / 255
    return img
    
def cropOrig(bRect, oimg):
    x, y, w, h = bRect
    pcropedImg = oimg[y:y+h, x:x+w]
    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]
    y2 = int(h1 / 10)
    x2 = int(w1 / 10)
    crop1 = pcropedImg[y1+y2:h1-y2, x1+x2:w1-x2]
    ix, iy, iw, ih = x+x2, y+y2, crop1.shape[1], crop1.shape[0]
    croppedImg = oimg[iy:iy+ih, ix:ix+iw]
    return croppedImg, pcropedImg

def overlayImage(croppedImg, pcropedImg):
    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]
    y2 = int(h1/10)
    x2 = int(w1/10)
    new_image = np.zeros((pcropedImg.shape[0], pcropedImg.shape[1], 3), np.uint8)
    new_image[:, 0:pcropedImg.shape[1]] = (255, 0, 0) # (B, G, R)
    new_image[ y1+y2:y1+y2+croppedImg.shape[0], x1+x2:x1+x2+croppedImg.shape[1]] = croppedImg
    return new_image

def kMeans_cluster(img):
    # For clustering the image using k-means, we first need to convert it into a 2-dimensional array
    # (H*W, N) N is channel = 3
    image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    # tweak the cluster size and see what happens to the Output
    kmeans = KMeans(n_clusters=2, random_state=0).fit(image_2D)
    clustOut = kmeans.cluster_centers_[kmeans.labels_]
    # Reshape back the image from 2D to 3D image
    clustered_3D = clustOut.reshape(img.shape[0], img.shape[1], img.shape[2])
    clusteredImg = np.uint8(clustered_3D*255)
    return clusteredImg

def getBoundingBox(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(len(contours))
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    return boundRect, contours, contours_poly, img

def drawCnt(bRect, contours, cntPoly, img):

    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)   


    paperbb = bRect

    for i in range(len(contours)):
      color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
      cv2.drawContours(drawing, cntPoly, i, color)
      #cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
              #(int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
    cv2.rectangle(drawing, (int(paperbb[0]), int(paperbb[1])), \
              (int(paperbb[0]+paperbb[2]), int(paperbb[1]+paperbb[3])), color, 2)
    
    return drawing

def edgeDetection(clusteredImage):
  #gray = cv2.cvtColor(hsvImage, cv2.COLOR_BGR2GRAY)
  edged1 = cv2.Canny(clusteredImage, 0, 255)
  edged = cv2.dilate(edged1, None, iterations=1)
  edged = cv2.erode(edged, None, iterations=1)
  return edged

def camera_capture():
    img_file_buffer = st.camera_input("Tirar foto")  # Captura de imagem da câmera

    if img_file_buffer is not None:  # Verificando dados da imagem

        # Lê o buffer do arquivo de imagem com o OpenCV:
        cv2_img = cv2.imdecode(np.frombuffer(img_file_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)

        # Pré-processa a imagem usando a função preprocess
        preprocessed_img = preprocess(cv2_img)

        # Exibe a imagem original
        st.image(cv2_img, caption='Foto Tirada', use_column_width=True)

        # Define o retângulo delimitador (x, y, largura, altura) para o recorte
        bRect = (100, 100, 200, 200)  # Você pode ajustar esses valores conforme necessário

        # Recorta e processa a imagem usando a função cropOrig
        cropped_img, pcropped_img = cropOrig(bRect, cv2_img) 

        # Pré-processa a imagem recortada e exibe-a
        preprocessed_cropped_img = preprocess(cropped_img)
        st.image(preprocessed_cropped_img, caption='Imagem Parcialmente Recortada Pré-Processada', use_column_width=True)

        # Aplica o algoritmo de clusterização K-Means à imagem pré-processada
        clustered_img = kMeans_cluster(preprocessed_cropped_img)

        # Exibe a imagem após a clusterização K-Means
        st.image(clustered_img, caption='Imagem Clusterizada', use_column_width=True)

        # Aplica a detecção de bordas à imagem clusterizada
        edged_img = edgeDetection(clustered_img)

        # Exibe a imagem com bordas detectadas
        st.image(edged_img, caption='Imagem com Bordas Detectadas', use_column_width=True)

        # Obtém informações dos contornos e desenha os contornos na imagem original  
        boundRect, contours, contours_poly, img_with_contours = getBoundingBox(edged_img)
        drawn_contours_img = drawCnt(boundRect[1], contours, contours_poly, img_with_contours)

        # Exibe a imagem com os contornos desenhados
        st.image(drawn_contours_img, caption='Imagem com Contornos Desenhados', use_column_width=True)

        # Corta a imagem clusterizada usando as informações de contorno
        cropped_img, pcropped_img = cropOrig(boundRect[1], clustered_img)

        # Exibe as duas imagens recortadas resultantes
        st.image(cropped_img, caption='Imagem Clusterizada Recortada', use_column_width=True)
        st.image(pcropped_img, caption='Imagem Parcialmente Clusterizada Recortada', use_column_width=True)

        # Cria uma nova imagem sobreposta
        new_img = overlayImage(cropped_img, pcropped_img)

        # Exibe a imagem sobreposta resultante
        st.image(new_img, caption='Imagem Sobreposta', use_column_width=True)

        # Aplica a detecção de bordas à imagem sobreposta
        fedged_img = edgeDetection(new_img)

        # Obtém informações dos contornos e desenha os contornos na imagem sobreposta
        fboundRect, fcontours, fcontours_poly, fimg_with_contours = getBoundingBox(fedged_img)
        fdrawn_contours_img = drawCnt(fboundRect[2], fcontours, fcontours_poly, fimg_with_contours)

        # Exibe a imagem sobreposta com contornos desenhados
        st.image(fdrawn_contours_img, caption='Imagem Sobreposta com Contornos Desenhados', use_column_width=True) 

def calcFeetSize(pcropedImg, fboundRect):
    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]
    y2 = int(h1 / 10)
    x2 = int(w1 / 10) 
    fh = y2 + fboundRect[2][3]
    fw = x2 + fboundRect[2][2]

    ph = pcropedImg.shape[0]
    pw = pcropedImg.shape[1]

    opw = 210
    oph = 297
 
    ofs = 0.0

    if fw > fh:
        ofs = (oph / pw) * fw
    else:
        ofs = (oph / ph) * fh

    return ofs
  
    st.write("Feet size (cm):", foot_size / 10) #Colocar a numeração padrão BR 
    
if __name__ == "__main__":
    camera_capture()



    