package image;

import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.Image;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.awt.image.BufferedImageOp;
import java.awt.image.Kernel;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import com.jhlabs.image.ConvolveFilter;
import com.jhlabs.image.GaussianFilter;
import com.jhlabs.image.MarbleFilter;
import com.jhlabs.image.MotionBlurFilter;
import com.jhlabs.image.NoiseFilter;
import com.jhlabs.image.RippleFilter;
import com.jhlabs.image.TwirlFilter;

public class DistortImage {
	public static final String urlPrefix = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=";
	
	public static InputStream getURLResults(String url, Map<String, Object> requestProperties, int retry) throws IOException
	{
		int tryCount = 0;
		
		while(true)
		{
			try
			{
				URL u = new URL(url);
				URLConnection yc = u.openConnection();
				yc.setConnectTimeout(1000);
				yc.setRequestProperty("User-Agent", "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36");				
				for(Map.Entry<String, Object> entry : requestProperties.entrySet())
				{
					yc.setRequestProperty(entry.getKey(), (String) entry.getValue());
				}
			
				Map<String, List<String>> map = yc.getHeaderFields();
				if(map.containsKey("Location"))
					return(getURLResults(map.get("Location").get(0), requestProperties, retry));
				return(yc.getInputStream());
			}
			catch (IOException ioe)
			{
				tryCount++;
				if(tryCount > retry) throw ioe;
			}
		}
	}
	
	public static List<String> getImageURLs(String synid) throws IOException
	{
		List<String> returnValue = new ArrayList<String>();
		BufferedReader br = new BufferedReader(new InputStreamReader(getURLResults(DistortImage.urlPrefix+synid, new HashMap<String, Object>(), 3)));
		String line;
		while((line = br.readLine()) != null)
			returnValue.add(line);
		
		return(returnValue);
	}
	
	public static Image getImage(String urlString) throws IOException
	{
		InputStream s = getURLResults(urlString, new HashMap<String, Object>(), 0);
		return(ImageIO.read(s));
	}
	
    public static void displayImage(Image img) throws IOException
    {
        ImageIcon icon=new ImageIcon(img);
        JFrame frame=new JFrame();
        frame.setLayout(new FlowLayout());
        frame.setSize(200,300);
        JLabel lbl=new JLabel();
        lbl.setIcon(icon);
        frame.add(lbl);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }
    
    public static Image blurImage(Image img) throws IOException
    {
    	return blurImage(img, 21);
    }

    // only works for odd kernelSize 
    public static Image blurImage(Image img, int kernelSize) throws IOException
    {
    	int kernelSizeSquared = kernelSize*kernelSize;
    	float[] matrix = new float[kernelSizeSquared];
		for (int j = 0; j < kernelSizeSquared; j++)
			matrix[j] = 1.0f/(float)kernelSizeSquared;
		
		ConvolveFilter cf = new ConvolveFilter(new Kernel(kernelSize, kernelSize, matrix));
		cf.setEdgeAction(2);
    	BufferedImageOp op = cf;
    	BufferedImage i2= op.filter((BufferedImage)img, null);
		return i2;
    }
    
    public static Image gaussianBlurImage(Image img) throws IOException
    {
    	return gaussianBlurImage(img, 10);
    }
    
    public static Image gaussianBlurImage(Image img, float radius) throws IOException
    {
		GaussianFilter gf = new GaussianFilter(radius);
    	BufferedImageOp op = gf;
    	BufferedImage i2= op.filter((BufferedImage)img, null);
		return i2;
    }
    
    public static Image motionBlurImage(Image img) throws IOException
    {
    	return motionBlurImage(img, 50, 90, 0, 0);
    }
    
    public static Image motionBlurImage(Image img, float distance, float angle, float rotation, float zoom) throws IOException
    {
    	MotionBlurFilter mf = new MotionBlurFilter();
    	mf.setAngle(angle);
    	mf.setDistance(distance);
    	mf.setRotation(rotation);
    	mf.setZoom(zoom);
    	BufferedImageOp op = mf;
    	BufferedImage i2= op.filter((BufferedImage)img, null);
		return i2;
    }
    
    public static Image marbleImage(Image img) throws IOException
    {
    	return marbleImage(img, 20, 20, 100, 5);
    }
    
    public static Image marbleImage(Image img, float xScale, float yScale, float amount, float turbulence) throws IOException
    {
    	MarbleFilter mf = new MarbleFilter();
    	mf.setXScale(xScale);
    	mf.setYScale(yScale);
    	mf.setAmount(amount);
    	mf.setTurbulence(turbulence);
    	BufferedImageOp op = mf;
    	BufferedImage i2= op.filter((BufferedImage)img, null);
		return i2;
    }
    
    public static Image rippleImage(Image img) throws IOException
    {
    	return rippleImage(img, 10, 10, 10, 10, 0);
    }
    
    public static Image rippleImage(Image img, float xAmplitude, float xWavelength, float yAmplitude, float yWavelength, int waveType) throws IOException
    {
    	RippleFilter mf = new RippleFilter();
    	mf.setXAmplitude(xAmplitude);
    	mf.setXWavelength(xWavelength);
    	mf.setYAmplitude(yAmplitude);
    	mf.setYWavelength(yWavelength);
    	mf.setWaveType(waveType);
    	BufferedImageOp op = mf;
    	BufferedImage i2= op.filter((BufferedImage)img, null);
		return i2;
    }
    
    public static Image twirlImage(Image img) throws IOException
    {
    	return twirlImage(img, 3, .5f, .5f, 250);
    }
    
    public static Image twirlImage(Image img, float angle, float centreX, float centreY, float radius) throws IOException
    {
    	TwirlFilter mf = new TwirlFilter();
    	mf.setAngle(angle);
    	mf.setCentreX(centreX);
    	mf.setCentreY(centreY);
    	mf.setRadius(radius);
    	BufferedImageOp op = mf;
    	BufferedImage i2= op.filter((BufferedImage)img, null);
		return i2;
    }
    
    public static Image noiseImage(Image img) throws IOException
    {
    	return noiseImage(img, 25, 0, false, 1);
    }
    
    public static Image noiseImage(Image img, int amount, int distribution, boolean monochrome, float density) throws IOException
    {
    	NoiseFilter mf = new NoiseFilter();
    	mf.setAmount(amount);
    	mf.setDistribution(distribution);
    	mf.setMonochrome(monochrome);
    	mf.setDensity(density);
    	BufferedImageOp op = mf;
    	BufferedImage i2= op.filter((BufferedImage)img, null);
		return i2;
    }
    
    // TODO: add random intensity of each distortion
    public static BufferedImage distortImage(Image i, int distortionClass) throws IOException
    {
    	BufferedImage i2 = null;
    	switch(distortionClass)
    	{
    		case 0:
    			i2 = (BufferedImage)gaussianBlurImage(i);
    			break;
    		case 1:
				i2 = (BufferedImage)motionBlurImage(i);
				break;
			case 2:
				i2 = (BufferedImage)noiseImage(i, 25, 0, false, 1);
				break;
			case 3:
				i2 = (BufferedImage)noiseImage(i, 25, 0, true, 1);
				break;
			case 4:
				i2 = (BufferedImage)marbleImage(i);
				break;
			case 5:
				i2 = (BufferedImage)rippleImage(i);
				break;
			case 6:
				i2 = (BufferedImage)twirlImage(i);
    			break;
    	}
    	return i2;
    }
    
    public static BufferedImage distortTwiceImage(Image i, int distortionClass1, int distortionClass2) throws IOException
    {
		BufferedImage i2 = (BufferedImage)distortImage(i, distortionClass1);
		return (BufferedImage)distortImage(i2, distortionClass2);
    }
    
    public static BufferedImage scaleImage(Image i, int smallSize) throws IOException
    {
    	BufferedImage bi = (BufferedImage) i;
    	final int w = bi.getWidth();
    	final int h = bi.getHeight();
    	
    	float scale = 1.F;
    	
    	if(h < w)
    		scale = (float)smallSize / (float) h;
    	else 
    		scale = (float)smallSize / (float) w;
    	
    	int newWidth = (int)((float)w * scale);
    	int newHeight = (int)((float)h * scale);
    	
    	BufferedImage scaledImage = new BufferedImage(newWidth,newHeight, BufferedImage.TYPE_INT_ARGB);
    	final AffineTransform at = AffineTransform.getScaleInstance(scale, scale);
    	final AffineTransformOp ato = new AffineTransformOp(at, AffineTransformOp.TYPE_BICUBIC);
    	scaledImage = ato.filter(bi, scaledImage);

//    	final Scale scaler = new Scale(2);
//    	BufferedImage scaledImage= scaler.apply(image);
//    	displayImage(bi);
//    	displayImage(scaledImage);
//    	System.out.println(scaledImage.getHeight()+","+scaledImage.getWidth());
    	return scaledImage;
    }
    
    public static BufferedImage cropImage(Image i, int size)
    {
    	BufferedImage bi = (BufferedImage) i;
    	final int w = bi.getWidth();
    	final int h = bi.getHeight();
    	
    	int xc = (w - size) / 2;
    	int yc = (h - size) / 2;
    	
    	 // Crop
        BufferedImage croppedImage = bi.getSubimage(
                        xc, 
                        yc,
                        size, // widht
                        size // height
        );
        return croppedImage;
    }
	
    static class ThreadWrapper implements Runnable {
    	private String s, className;
    	private int imageNumber;
    	private Random r;
    	
    	public ThreadWrapper(String s, int imageNumber, Random r, String className)
    	{
    		this.s = s;
    		this.imageNumber = imageNumber;
    		this.r = r;
    		this.className = className;
    	}
    	
    	public void run() 
    	{
//    		long startDownload = System.currentTimeMillis();
			Image i = null;
			try {
				i = getImage(s);
				if(i==null) return;
				i = scaleImage(i, 224);
				i = cropImage(i, 224);
				if(i==null) return;
			} catch(Exception e)
			{
				e.printStackTrace();
				return;
			}
//			long endDownload = System.currentTimeMillis();
			//						displayImage(i);
//			download += endDownload - startDownload;
			
			String formatted = String.format("%08d", imageNumber);
			System.out.println(imageNumber+","+s);

			// Distortions
			// 0. Gaussian Blur "smooth blur"
			// 1. Motion Blur
			// 2. Gaussian Noise non-monochrome
			// 3. Gaussian Noise monochrome
			// 4. Marble
			// 5. Ripple
			// 6. Twirl

			for(int distortionClass = 0; distortionClass < 8; distortionClass++)
			{
//				long startProcessing = System.currentTimeMillis();
				BufferedImage i2 = null;
				String distortionClassLabel = "";
				if(distortionClass == 7)
				{
					int first = r.nextInt(7);
					int second = r.nextInt(7);
					while(second == first)
						second = r.nextInt(7);
					try {
						i2 = distortTwiceImage(i, first, second);
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
						return;
					}
					distortionClassLabel = ""+(first+1)+""+(second+1);
				}
				else
				{
					try {
						i2 = distortImage(i, distortionClass);
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
						return;
					}
					distortionClassLabel = ""+(distortionClass+1);
				}
				//				displayImage(i2);
//				long endProcessing = System.currentTimeMillis();
//				processing += endProcessing - startProcessing;
				
//				long startDisk = System.currentTimeMillis();
				
				File outputfile = new File("C:\\out\\"+formatted+"."+distortionClassLabel+"."+className+".jpg");
				try {
					BufferedImage bi = (BufferedImage)i2;
					BufferedImage result = new BufferedImage(
			                    bi.getWidth(),
			                    bi.getHeight(),
			                    BufferedImage.TYPE_INT_RGB);
					result.createGraphics().drawImage(bi, 0, 0, Color.WHITE, null);
					ImageIO.write(result, "jpg", outputfile);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
					return;
				}
//				long endDisk = System.currentTimeMillis();
//				disk += endDisk - startDisk;
			}
			File outputfile = new File("C:\\out\\"+formatted+".0."+className+".jpg");
			try {
				BufferedImage bi = (BufferedImage)i;
				BufferedImage result = new BufferedImage(
		                    bi.getWidth(),
		                    bi.getHeight(),
		                    BufferedImage.TYPE_INT_RGB);
				result.createGraphics().drawImage(bi, 0, 0, Color.WHITE, null);
				ImageIO.write(result, "jpg", outputfile);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				return;
			}
    	}
    }
    
	public static void main(String[] args) throws IOException
	{
		// from https://towardsdatascience.com/how-to-scrape-the-imagenet-f309e02de1f4
		// https://github.com/mf1024/ImageNet-datasets-downloader
		File file = new File("data\\classes_in_imagenet.csv");
		int startLine = 1870;
		int imageNumber=193021;
		Random r = new Random(System.currentTimeMillis());
		  
		BufferedReader br = new BufferedReader(new FileReader(file)); 

		String st;
		long download = 0, processing = 0, disk = 0, start = 0;

		
		int lineNumber = 0;
		
		ExecutorService executor = Executors.newFixedThreadPool(100);
		
		while ((st = br.readLine()) != null)
		{
			if(st.startsWith("synid")) continue;
			lineNumber++;
			if(lineNumber < startLine) continue;
						
			String[] words = st.split(",");
			String synid = words[0];
			String className = words[1];
			
			List<String> list = getImageURLs(synid);
			
			start = System.currentTimeMillis();
			for(String s : list)
			{
				if(s.length()==0) continue;
				System.out.println("LINE:"+lineNumber+","+s);
				imageNumber++;
				if(imageNumber > 1000000) break;
				Runnable worker = new DistortImage.ThreadWrapper(s, imageNumber, r, className);
				executor.execute(worker);
			}
		}
		br.close();
		executor.shutdown();
		while (!executor.isTerminated() ) {
		}
		long end = System.currentTimeMillis();
		System.out.println((end-start)/1000);
//		System.out.println(download/1000);
//		System.out.println(processing/1000);
//		System.out.println(disk/1000);
	}
}
