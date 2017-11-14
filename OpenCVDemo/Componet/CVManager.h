//
//  cvManager.h
//  OpenCVDemo
//
//  Created by ice.hu on 2017/11/7.
//  Copyright © 2017年 ice. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreMotion/CoreMotion.h>


@interface CVManager: NSObject
+ (CVManager *)shared;

-(UIImage *) stitchWithOpenCV:(NSArray *) images;
-(UIImage *) startTest:(UIImage *)pimage1 image2:(UIImage *)pimage2;

@end
