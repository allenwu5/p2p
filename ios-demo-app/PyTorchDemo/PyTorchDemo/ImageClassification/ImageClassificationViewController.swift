import AVFoundation
import UIKit
import Vision
class ImageClassificationViewController: ViewController {
    @IBOutlet var cameraView: CameraPreviewView!
    @IBOutlet var bottomView: ImageClassificationResultView!
    @IBOutlet var benchmarkLabel: UILabel!
    @IBOutlet var indicator: UIActivityIndicatorView!
    private var predictor = ImagePredictor()
    private var cameraController = CameraController()
    private let delayMs: Double = 500
    private var prevTimestampMs: Double = 0.0
    private var drawings: [CAShapeLayer] = []
    override func viewDidLoad() {
        super.viewDidLoad()
        bottomView.config(resultCount: 3)
        cameraController.configPreviewLayer(cameraView)
        cameraController.videoCaptureCompletionBlock = { [weak self] buffer, _pixelBuffer, previewView, error in
            guard let strongSelf = self else { return }
            if error != nil {
                strongSelf.showAlert(error)
                return
            }
            
            self!.detectFace(in: _pixelBuffer!, in:previewView!)
            
            guard let pixelBuffer = buffer else { return }
            let currentTimestamp = CACurrentMediaTime()
            if (currentTimestamp - strongSelf.prevTimestampMs) * 1000 <= strongSelf.delayMs { return }
            strongSelf.prevTimestampMs = currentTimestamp
            if let results = try? strongSelf.predictor.predict(pixelBuffer, resultCount: 3) {
                DispatchQueue.main.async {
                    strongSelf.indicator.isHidden = true
                    strongSelf.bottomView.isHidden = false
                    strongSelf.benchmarkLabel.isHidden = false
                    strongSelf.benchmarkLabel.text = String(format: "%.2fms", results.1)
                    strongSelf.bottomView.update(results: results.0)
                }
            }
        }
    }
    
    // https://medium.com/onfido-tech/live-face-tracking-on-ios-using-vision-framework-adf8a1799233
    private func detectFace(in image:CVPixelBuffer, in previewView:CameraPreviewView) {
        let faceDetectionRequest = VNDetectFaceLandmarksRequest(completionHandler: { (request: VNRequest, error: Error?) in
            DispatchQueue.main.async {
                if let results = request.results as? [VNFaceObservation] {
                    self.handleFaceDetectionResults(results, previewView: previewView)
                } else {
                    self.clearDrawings()
                }
            }
        })
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: image, orientation: .leftMirrored, options: [:])
        try? imageRequestHandler.perform([faceDetectionRequest])
    }
    
    private func handleFaceDetectionResults(_ observedFaces: [VNFaceObservation], previewView:CameraPreviewView) {
        self.clearDrawings()
        let facesBoundingBoxes: [CAShapeLayer] = observedFaces.flatMap({ (observedFace: VNFaceObservation) -> [CAShapeLayer] in
            var faceBoundingBoxOnScreen = previewView.previewLayer.layerRectConverted(fromMetadataOutputRect: observedFace.boundingBox)
            // FIXME: why dy = 22 ?
            faceBoundingBoxOnScreen = faceBoundingBoxOnScreen.offsetBy(dx: 0, dy: 22)
            let faceBoundingBoxPath = CGPath(rect: faceBoundingBoxOnScreen, transform: nil)
            let faceBoundingBoxShape = CAShapeLayer()
            faceBoundingBoxShape.path = faceBoundingBoxPath
            faceBoundingBoxShape.fillColor = UIColor.clear.cgColor
            faceBoundingBoxShape.strokeColor = UIColor.green.cgColor
            var newDrawings = [CAShapeLayer]()
            newDrawings.append(faceBoundingBoxShape)
            if let landmarks = observedFace.landmarks {
                newDrawings = newDrawings + self.drawFaceFeatures(landmarks, screenBoundingBox: faceBoundingBoxOnScreen)
            }
            return newDrawings
        })
        facesBoundingBoxes.forEach({ faceBoundingBox in self.view.layer.addSublayer(faceBoundingBox) })
        self.drawings = facesBoundingBoxes
    }
    
    private func clearDrawings() {
        self.drawings.forEach({ drawing in drawing.removeFromSuperlayer() })
    }
    
    private func drawFaceFeatures(_ landmarks: VNFaceLandmarks2D, screenBoundingBox: CGRect) -> [CAShapeLayer] {
        var faceFeaturesDrawings: [CAShapeLayer] = []
        if let leftEye = landmarks.leftEye {
            let eyeDrawing = self.drawEye(leftEye, screenBoundingBox: screenBoundingBox)
            faceFeaturesDrawings.append(eyeDrawing)
        }
        if let rightEye = landmarks.rightEye {
            let eyeDrawing = self.drawEye(rightEye, screenBoundingBox: screenBoundingBox)
            faceFeaturesDrawings.append(eyeDrawing)
        }
        // draw other face features here
        if let leftEye = landmarks.leftEye, let rightEye = landmarks.rightEye {
            let eyeMaskDrawing = self.drawEyeMask(leftEye, rightEye, screenBoundingBox: screenBoundingBox)
            faceFeaturesDrawings.append(eyeMaskDrawing)
        }
        return faceFeaturesDrawings
    }
    
    private func drawEye(_ eye: VNFaceLandmarkRegion2D, screenBoundingBox: CGRect) -> CAShapeLayer {
        let eyePath = CGMutablePath()
        let eyePathPoints = eye.normalizedPoints
            .map({ eyePoint in
                CGPoint(
                    x: (1-eyePoint.y) * screenBoundingBox.height + screenBoundingBox.origin.x,
                    y: eyePoint.x * screenBoundingBox.width + screenBoundingBox.origin.y)
             })
        eyePath.addLines(between: eyePathPoints)
        eyePath.closeSubpath()
        let eyeDrawing = CAShapeLayer()
        eyeDrawing.path = eyePath
        eyeDrawing.fillColor = UIColor.clear.cgColor
        eyeDrawing.strokeColor = UIColor.green.cgColor
        
        return eyeDrawing
    }
    
    private func drawEyeMask(_ leftEye: VNFaceLandmarkRegion2D, _ rightEye: VNFaceLandmarkRegion2D, screenBoundingBox: CGRect) -> CAShapeLayer {
        let eyePath = CGMutablePath()
        var eyePathPoints = leftEye.normalizedPoints
            .map({ eyePoint in
                CGPoint(
                    x: (1-eyePoint.y) * screenBoundingBox.height + screenBoundingBox.origin.x,
                    y: eyePoint.x * screenBoundingBox.width + screenBoundingBox.origin.y)
             })
        eyePath.addLines(between: eyePathPoints)
        
        eyePathPoints = rightEye.normalizedPoints
            .map({ eyePoint in
                CGPoint(
                    x: (1-eyePoint.y) * screenBoundingBox.height + screenBoundingBox.origin.x,
                    y: eyePoint.x * screenBoundingBox.width + screenBoundingBox.origin.y)
             })
        eyePath.addLines(between: eyePathPoints)
        eyePath.closeSubpath()
        
        let rect = eyePath.boundingBoxOfPath
        let path = CGMutablePath(rect: rect, transform: nil)
        
        let eyeDrawing = CAShapeLayer()
        eyeDrawing.path = path
        eyeDrawing.fillColor = UIColor.black.cgColor
        eyeDrawing.strokeColor = UIColor.white.cgColor
        eyeDrawing.lineWidth = 5.0
        
        return eyeDrawing
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        navigationController?.setNavigationBarHidden(true, animated: false)
        cameraController.startSession()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        cameraController.stopSession()
    }

    @IBAction func onInfoBtnClicked(_: Any) {
        VisionModelCard.show()
    }

    @IBAction func onBackClicked(_: Any) {
        navigationController?.popViewController(animated: true)
    }
}
