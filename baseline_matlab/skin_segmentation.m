clear all;
close all;
clc;

% --- KONFİGÜRASYON ---
config.min_area = 5000; % Arka plan gürültüsünü silmek için daha yüksek eşik
config.keep_largest_n = 3; % Sadece en büyük N bölgeyi tut (Örn: Yüz ve iki el)
config.morph_close_size = 15; % Yüzdeki boşlukları doldurmak için büyük yapısal eleman
config.show_stats_every = 30; % Log sıklığı

fprintf('\n=== CİLT RENGİ SEGMENTASYONU ANALİZ SİSTEMİ ===\n');
fprintf('1. Fotoğraf İşle (Dosyadan Seç)\n');
fprintf('2. Canlı Kamera (3 Panelli İzleme)\n');
mode = input('Seçiminiz (1-2): ');

switch mode
    case 1
        [fn, pn] = uigetfile({'*.jpg;*.png;*.bmp;*.jpeg'});
        if ischar(fn)
            img = imread(fullfile(pn, fn));
            processSingleImage(img, config);
        end
    case 2
        processWebcam(config);
    otherwise
        fprintf('Geçersiz seçim.\n');
end

%% ==================== CANLI KAMERA ========================
function processWebcam(config)
    try
        cam = webcam;
        hFig = figure('Name', 'Cilt Analizi', 'NumberTitle', 'off', ...
                      'Position', [50, 100, 1600, 600], 'Color', [0.1 0.1 0.1]);
        
        frameCount = 0;
        startTime = tic;
        
        fprintf('\nCanlı yayın başladı. Durdurmak için pencereyi kapatın.\n');
        
        while ishandle(hFig)
            tFrame = tic;
            frame = snapshot(cam);
            
            % Gelişmiş İşleme
            [mask, numRegions, contours] = highPrecisionSkinDetection(frame, config);
            
            % Görselleştirme
            subplot(1, 3, 1); imshow(frame); 
            title('SOL: Canlı Kamera', 'FontSize', 12, 'Color', 'w');
            
            subplot(1, 3, 2); imshow(mask); 
            title('ORTA: Cilt Maskesi (Temizlenmiş)', 'FontSize', 12, 'Color', 'w');
            
            subplot(1, 3, 3); imshow(frame); hold on;
            for k = 1:length(contours)
                boundary = contours{k};
                plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 2.5);
            end
            title('SAĞ: Tespitler', 'FontSize', 12, 'Color', 'w');
            hold off;
            
            % İstatistikler
            procTime = toc(tFrame);
            frameCount = frameCount + 1;
            avgFps = frameCount / toc(startTime);
            
            sgtitle(sprintf('FPS: %.1f | Frame: %d | Bölgeler: %d', ...
                    1/procTime, frameCount, numRegions), 'FontSize', 14, 'Color', [0 1 0], 'FontWeight', 'bold');
            
            if mod(frameCount, config.show_stats_every) == 0
                fprintf('LOG [%s] - FPS: %.1f, Tespit: %d\n', datestr(now, 'HH:MM:SS'), avgFps, numRegions);
            end
            
            drawnow limitrate;
        end
    catch ME
        fprintf('\nHata: %s\n', ME.message);
        fprintf('İpucu: Webcam Toolbox yüklü mü? "webcamlist" komutunu deneyin.\n');
    end
end

%% ==================== YÜKSEK HASSASİYETLİ SEGMENTASYON =================
function [finalMask, numRegions, contours] = highPrecisionSkinDetection(img, config)
    % 1. Renk Uzayı Dönüşümleri
    imgYCbCr = rgb2ycbcr(img);
    imgHSV = rgb2hsv(img);
    
    % 2. Agresif Eşikleme (Arka planı elemek için daraltılmış aralıklar)
    Cb = imgYCbCr(:,:,2);
    Cr = imgYCbCr(:,:,3);
    H = imgHSV(:,:,1);
    S = imgHSV(:,:,2);
    
    % Cb aralığını daraltarak bej/sarı arka planları eliyoruz.
    % Cr aralığını hafifçe açarak kırmızılığı (dudak, yanak) yakalıyoruz.
    mask = (Cb >= 80 & Cb <= 125) & (Cr >= 135 & Cr <= 180) & ...
           (H >= 0.0 & H <= 0.1) & (S >= 0.1 & S <= 0.7);
       
    % 3. Gelişmiş Morfolojik Temizlik (Sıralama Önemli!)
    % Adım A: Yüzdeki büyük boşlukları (gözlük, sakal) kapatmak için 'Closing'
    se = strel('disk', config.morph_close_size);
    mask = imclose(mask, se);
    
    % Adım B: İçteki tüm delikleri doldur (Tam yüz maskesi için)
    mask = imfill(mask, 'holes');
    
    % Adım C: Küçük arka plan gürültülerini sil (Area Opening)
    mask = bwareaopen(mask, config.min_area);
    
   % 4. Gelişmiş ve Katı Bölge Seçimi
    [L, num] = bwlabel(mask);
    if num > 0
        % Geometrik özellikleri çıkar
        stats = regionprops(L, 'Area', 'Eccentricity', 'Extent', 'PixelIdxList');
        
        % Kriterlere göre filtreleme (Logical Indexing)
        % 1. Eccentricity < 0.90: Çok ince uzun nesneleri (fayans çizgisi) eler
        % 2. Extent < 0.80: Tam dikdörtgen/kare nesneleri (dolap kapağı yansıması) eler
        % 3. Area > config.min_area: Belirlenen minimum boyuttan küçükleri eler
        validIdx = ([stats.Eccentricity] < 0.90) & ...
                   ([stats.Extent] < 0.80) & ...
                   ([stats.Area] > config.min_area);
        
        if any(validIdx)
            % Sadece geçerli olan bölgelerin istatistiklerini al
            filteredStats = stats(validIdx);
            [~, sortedIdx] = sort([filteredStats.Area], 'descend');
            
            % Sadece en büyük N taneyi (veya daha azını) tut
            nToKeep = min(length(sortedIdx), config.keep_largest_n);
            finalMask = false(size(mask));
            
            % En büyük bölgenin alanını referans al (Hiyerarşik Filtre)
            maxArea = filteredStats(sortedIdx(1)).Area;
            
            for i = 1:nToKeep
                currIdx = sortedIdx(i);
                % Eğer bölge, en büyük bölgenin %20'sinden küçükse 'uzuv' olamaz, ele.
                if filteredStats(currIdx).Area > (maxArea * 0.20)
                    finalMask(filteredStats(currIdx).PixelIdxList) = true;
                end
            end
        else
            finalMask = false(size(mask));
        end
    else
        finalMask = mask;
    end
    
    % 5. Nihai Analiz
    [~, numRegions] = bwlabel(finalMask);
    contours = bwboundaries(finalMask, 'noholes');
end

%% ==================== FOTOĞRAF İŞLEME ===============================
function processSingleImage(img, config)
    [mask, numRegions, contours] = highPrecisionSkinDetection(img, config);
    
    figure('Name', 'Fotoğraf Analizi', 'Position', [100, 100, 1200, 500]);
    subplot(1,2,1); imshow(img); title('Orijinal');
    subplot(1,2,2); imshow(img); hold on;
    for k = 1:length(contours)
        boundary = contours{k};
        plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 2.5);
    end
    title(['Tespit: ' num2str(numRegions) ' Bölge (En Büyük ' num2str(config.keep_largest_n) ')']);
end