# IOS - OpenCV - Demo

<!--markdown-->
<!-- toc -->

# 前言

This is a demo for image stitching use swift & object-c & OpenCV 3.2 .

Hope it's helpful for anyone want to use OpenCV in IOS.

这是一个使用 Swift 开发的图片拼接 demo , 利用 object-c 整合 OpenCV 3.2

希望对在 IOS 中使用 OpenCV 的朋友有点帮助

# 使用

修改循环中的数字,对应文件夹中的图片名字,也可以加入新的图片.
```Swift
for i in 0...17{
    if let image = UIImage(named:"\(i).jpg") {
         images.append(image)
     }
}

```

此方法为 使用OpenCV 自带的方法实现 图片拼接
```Objc
-(UIImage *) stitchWithOpenCV:(NSArray *) images;
```

此方法为 根据Madcola 前辈的 帖子把 他使用的 opencv2 转 成 opencv3.2 实现的图片拼接.
```Objc
-(UIImage *) startTest:(UIImage *)pimage1 image2:(UIImage *)pimage2;
```

# 资料

以下自学过程中的搜集的一些资料

[open cv 官网](https://opencv.org/)

[Madcola 的 博客](http://www.cnblogs.com/skyfsm/p/7411961.html)

[如何 openCV 3 后使用 SIFT and SURF 算法](https://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/)
