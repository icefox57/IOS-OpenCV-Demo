platform :ios, ’10.0’
use_frameworks!

target ‘OpenCVDemo’ do
    #base


    #component
    pod 'OpenCV', '~> 3.2.0'


end

post_install do |installer|
    installer.pods_project.targets.each do |target|
        target.build_configurations.each do |config|
            config.build_settings['CONFIGURATION_BUILD_DIR'] = '$PODS_CONFIGURATION_BUILD_DIR'
            config.build_settings['SWIFT_VERSION'] = '4.0'
        end
    end
end
