//
//  ViewController.swift
//  OpenCVDemo
//
//  Created by ice.hu on 2017/11/7.
//  Copyright © 2017年 ice. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    @IBOutlet weak var imgvOutput: UIImageView!
    @IBOutlet weak var imgv1: UIImageView!
    @IBOutlet weak var imgv2: UIImageView!
    let cvmanager = CVManager.shared()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        imgv1.image = UIImage(named:"0.jpg")
        imgv2.image = UIImage(named:"5.jpg")

    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    override func viewDidAppear(_ animated: Bool) {
        
    }
    
    @IBAction func testClicked(_ sender: Any) {
        imgvOutput.image =  cvmanager?.startTest(imgv2.image!, image2: imgv1.image!)
    }
    
    @IBAction func stitchClicked(_ sender: Any) {
        var images:[UIImage] = []
        for i in 0...17{
            if let image = UIImage(named:"\(i).jpg") {
                images.append(image)
            }
        }
        
        imgvOutput.image = cvmanager?.stitch(withOpenCV: images)
    }
    
}

