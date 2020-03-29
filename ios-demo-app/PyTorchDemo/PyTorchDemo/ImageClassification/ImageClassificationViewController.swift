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
    
//    https://medium.com/onfido-tech/live-face-tracking-on-ios-using-vision-framework-adf8a1799233
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
        let facesBoundingBoxes: [CAShapeLayer] = observedFaces.map({ (observedFace: VNFaceObservation) -> CAShapeLayer in
            let faceBoundingBoxOnScreen = previewView.previewLayer.layerRectConverted(fromMetadataOutputRect: observedFace.boundingBox)
            let faceBoundingBoxPath = CGPath(rect: faceBoundingBoxOnScreen, transform: nil)
            let faceBoundingBoxShape = CAShapeLayer()
            faceBoundingBoxShape.path = faceBoundingBoxPath
            faceBoundingBoxShape.fillColor = UIColor.clear.cgColor
            faceBoundingBoxShape.strokeColor = UIColor.green.cgColor
            return faceBoundingBoxShape
        })
        facesBoundingBoxes.forEach({ faceBoundingBox in self.view.layer.addSublayer(faceBoundingBox) })
        self.drawings = facesBoundingBoxes
    }
    private func clearDrawings() {
        self.drawings.forEach({ drawing in drawing.removeFromSuperlayer() })
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
