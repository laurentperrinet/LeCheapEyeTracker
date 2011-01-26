import cv

# # taking a snapshot
# 
# capture = cv.CaptureFromCAM(1)
# # check if capture device is OK
# downsize = 1
# img = cv.Queryimg(capture)
# cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_img_HEIGHT, img.height / downsize)
# cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_img_WIDTH, img.width / downsize)
# cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FORMAT, cv.IPL_DEPTH_32F)
# 
# img = cv.Queryimg(capture)
# 
# cv.SaveImage("snapshot.png", img)
# 
# applying SOBEL or LAPLACE
img = cv.LoadImage('snapshot.png', cv.CV_LOAD_IMAGE_COLOR)

img32F = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_32F, 3)
cv.ConvertScale(img, img32F)
# 
# imgGS_32F = cv.CreateImage (cv.GetSize(img), cv.IPL_DEPTH_32F, 1)
# cv.CvtColor(img32F, imgGS_32F, cv.CV_RGB2GRAY)
# 
# imgGS = cv.CreateImage (cv.GetSize(img), cv.IPL_DEPTH_8U, 1)
# cv.ConvertScale(imgGS_32F, imgGS)
# 


cv.ShowImage("Snapshot", img)
planes = [cv.CreateImage(cv.GetSize(img), 8, 1) for i in range(3)]
laplace = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_32F, 1)
colorlaplace = cv.CreateImage((img.width, img.height), 8, 3)

cv.Split(img, planes[0], planes[1], planes[2], None)
for plane in planes:
    cv.Laplace(plane, laplace, 3)
    cv.ConvertScaleAbs(laplace, plane, 1, 0)

cv.Merge(planes[0], planes[1], planes[2], None, colorlaplace)

cv.ShowImage("Color Laplacian", colorlaplace)


gray = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_32F, 1)
cv.CvtColor(img32F, gray, cv.CV_BGR2GRAY)
laplace = cv.CreateImage((img.width, img.height), cv.IPL_DEPTH_32F, 1)
cv.Laplace(gray, laplace, 3)
cv.ShowImage("Laplacian", laplace)


# result = cv.CreateImage(cv.GetSize(img), 8, 1)
# cv.Laplace(gray, result)
# cv.Smooth(gray, gray, smoothtype=cv.CV_GAUSSIAN, param1=7, param2=0, param3=0, param4=0)
# 
# cv.EqualizeHist(gray, gray)
# fe = fit_ellipse(gray, 90)


