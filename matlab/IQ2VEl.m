% IQ = readNPY('IQData1_10.npy');
% % % Commented and cleaned with Prof. Karla P. Mercado-Shekhar
%Check if data save properly by viewing B-mode image
%% Construct IQ data from multiple plane-wave acquisitions
% IBuffer and QBuffer are assumed to contain in-phase and quadrature data.
% After combining them, we reshape into:
% IQData: [nRows, nCols, na, nFocusX]
% where:
%   nRows   = axial samples (depth)
%   nCols   = lateral samples
% Combine I/Q buffers into complex IQ representation
% load('20260128T175321.mat')
IQData  = complex(IBuffer, QBuffer);

% Remove singleton dimensions (commonly from page/frame indexing)
% Resulting IQDataS should be [nRows, nCols, nPages]
IQDataS = squeeze(IQData);

% Reshape into [nRows, nCols, na, nFocusX]
IQData  = reshape(IQDataS, size(IQData,1), size(IQData,2), na, length(SWIFocusX));

%% Determine the acquisition ROI based on non-zero IQ samples
% We detect where IQData is non-zero for the first angle and first focus index.
% NOTE: This assumes that non-zero entries correspond to valid imaging region.

[ROIrows, ROIcols] = find(abs(IQData(:,:,1,1)));

% ROI extents (axial and lateral bounds)
minROIrow = min(ROIrows);
maxROIrow = max(ROIrows);
minROIcol = min(ROIcols);
maxROIcol = max(ROIcols);

% ROI indices used for later processing
ROI_acq = 1:50;                   % acquisition
ROI_ax  = minROIrow:maxROIrow;    % axial index range
ROI_lat = minROIcol:maxROIcol;    % lateral index range

%% Retrieve acoustic and transducer parameters
if exist('speedOfSound', 'var')
    c = speedOfSound;                          % speed of sound [m/s]
else
    c = Resource.Parameters.speedOfSound;      % speed of sound [m/s]
end

% Transmit (imaging) center frequency
f_c = Trans.frequency * 1e6;                   % center frequency [Hz]

%% Plot B-mode image (log-compressed, dB)
% This block:
% 1) computes magnitude of IQ data for one frame/page,
% 2) normalizes to its maximum,
% 3) converts to dB (log compression),
% 4) displays the ROI using lateral/axial axes,
% 5) saves the figure in .fig and .jpg formats.

% --- Select one page/frame (here: page 3) ---
IQmag = abs(IQDataS(:,:,3));                 % magnitude image (linear scale)

% --- Normalize by global maximum to map into [0,1] before dB conversion ---
IQmagNorm = IQmag / max(IQmag(:));           % robust max using vectorization

% --- Convert to decibels (log compression) ---
Flashcomp = mag2db(squeeze(IQmagNorm));      % dB image (typically <= 0 dB)

% --- Display ROI on the given axes ---
figure(1);
imagesc(axisWidth, axisDepth, Flashcomp(ROI_ax, ROI_lat));  % plot only ROI
colormap gray
axis image                                     % equal aspect ratio

% --- Labels and title ---
xlabel('Lateral distance from center (mm)', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Axial depth (mm)',                  'FontSize', 20, 'FontWeight', 'bold');
title('B-mode image ROI (dB)');

% --- Display range and colorbar ---
clim([-40 0]);                                % show [-40, 0] dB range
colorbar;

% --- Save and close ---
savefig('Z_FlashDb.fig');
saveas(gcf, 'Z_FlashDb.jpg');
close(figure(1));

%% Compute axial displacement using 2-D Loupas autocorrelator (Loupas et al., 1995)
% This block estimates axial displacement from IQ data using a 2-D
% autocorrelation approach (temporal + spatial) as in:
%   - Loupas et al., 1995
%   - Maksuti et al., 2016 (related usage in elastography/ultrasound tracking)

h = waitbar(0, 'Computing Axial Displacement (Step 1 of 4)');

% NOTE (your comment, kept): sampling frequency is 5.208 MHz*4 => sampled every
% 1540/(5.208 MHz*4) = 0.074 mm (~1/4th wavelength)

% Loupas autocorrelator kernel sizes
M = 6;   % range gate length (axial samples); ~1.5 wavelengths (as you noted: low jitter)
N = 3;   % ensemble length (number of frames/acquisitions per kernel)

% Extract IQ over the acquisition dimension
IQDataS=squeeze(IQData(:,:,1:na));
IQS=IQDataS(ROI_ax,ROI_lat,ROI_acq);
for nacq_i=2:length(ROI_acq)  %2:30
    % compute displacement using 2-D autocorrelator (Loupas et al., 1995 and Maksuti et al., 2016)
    nacq = ROI_acq(nacq_i);  %2:30
     % Skip if not enough frames ahead for N
    if nacq + N - 1 > size(IQDataS, 3)
        continue; 
    end
    for ax_i = 2:length(ROI_ax) 
        ax = ROI_ax(ax_i); 
        if ax + M - 1 > size(IQDataS, 1)
            continue;
        end
        I = real(IQDataS(ax:ax+M-1, ROI_lat, nacq:nacq+N-1));  %saved separately (1:160 lateral locations, 20 21 22)
        Q = imag(IQDataS(ax:ax+M-1, ROI_lat, nacq:nacq+N-1));  %
        % ------------------------------------------------------------
        % Loupas 2-D autocorrelator terms
        % Temporal correlation between consecutive frames (n -> n+1)
        % Spatial correlation between consecutive axial samples (m -> m+1)
        % ------------------------------------------------------------
        uu = Q(1:M-1,:,1:N-1).*I(1:M-1,:,2:N) - I(1:M-1,:,1:N-1).*Q(1:M-1,:,2:N);  % (1:M-1,:,1:N-1)=(1:5,:,1:2)
        ud = I(1:M-1,:,1:N-1).*I(1:M-1,:,2:N) + Q(1:M-1,:,1:N-1).*Q(1:M-1,:,2:N);
        du = Q(1:M-1,:,1:N-1).*I(2:M,:,1:N-1) - I(1:M-1,:,1:N-1).*Q(2:M,:,1:N-1);
        dd = I(1:M-1,:,1:N-1).*I(2:M,:,1:N-1) + Q(1:M-1,:,1:N-1).*Q(2:M,:,1:N-1);
        uu_sum = squeeze(sum(squeeze(sum(uu,1)),2));
        ud_sum = squeeze(sum(squeeze(sum(ud,1)),2));
        du_sum = squeeze(sum(squeeze(sum(du,1)),2));
        dd_sum = squeeze(sum(squeeze(sum(dd,1)),2));
        disp(ax_i,:,nacq_i) = c/(4*pi*f_c).*atan(uu_sum./ud_sum)./ ...
            (1+atan(du_sum./dd_sum)./(2*pi))*1e6; % displacement in microns
    end
end
close(h); 
display('Finished computing tissue displacements.')
%% Smooth displacement data (spatial directions only) using 2D median filter
h = waitbar(0, 'Smoothing displacement data using 2D median filter (Step 2 of 4)');

for nacq_i = 2:size(disp,3)
    % Transmit frequency = 7.6 MHz
    % kernel size = 0.6 mm x 0.6 mm [3*lambda]
    % axial pixel distance = 0.0507 mm -> axial pixel distance x 12 = 0.6084 mm
    % lateral pixel distance = 0.1013 mm -> lateral pixel distance x 6 = 0.6078 mm

    temp = medfilt2(squeeze(disp(:,:,nacq_i)),[12 6]);
    %remove rows with NaN values bec. medfilt2 might be adding some extra pixels
    medfiltdisp(:,:,nacq_i)= temp(~isnan(temp(:,1)),:);
end
axisDepth=axisDepth(1:size(medfiltdisp,1));

display('Finished median filtering tissue displacement data.')
close(h);

%% Perform 10x slow-time interpolation of displacement data at each spatial location
% Increase sampling frequency
interpfactor = 10;
% increasing interpfactor further > 10 doesn't improve smoothing of displacement data
h = waitbar(0, 'Performing Data Interpolation (Step 3 of 4)');
clear interpfiltdisp
for ax_i = 1:size(medfiltdisp,1)
    for lat_i = 1:size(medfiltdisp,2)
        AA=squeeze(medfiltdisp(ax_i,lat_i,:));
        AA(isnan(AA)) = 0;
        interpdisp(ax_i,lat_i,:)=interp1(1:1:size(medfiltdisp,3),...
            AA, 1:1/interpfactor:size(medfiltdisp,3),'spline');
    end
end
close(h);

%%
% Compute material velocity over slow time
% Differentiate displacement vs time profiles using finite-difference method
frame_time = DetectPRP/1e6; % Pulse repetition period for shear wave detection
slowtimeaxis = (Timeafterpush*1e-6 : frame_time: Timeafterpush*1e-6 + frame_time * (size(disp,3 ) -1) );
taxis = (Timeafterpush*1e-6 : frame_time/interpfactor : Timeafterpush*1e-6 +frame_time * (size(disp,3 ) -1) ); %seconds

h = waitbar(0, 'Computing material velocity over slow time (Step 4 of 4)');

clear vel
for nacq_i=2:size(interpdisp,3)
    for ax_i = 1:size(interpdisp,1)
        vel(ax_i,:,nacq_i)=(interpdisp(ax_i,:,nacq_i)-interpdisp(ax_i,:,nacq_i-1))./(frame_time/interpfactor*1e3);% mm/s
    end
end

close(h);
%% Bandpass filter velocity data between 10 Hz and 800 Hz
T=(taxis(2)-taxis(1));%s
Fs=(1/T);% sampling frequency (Hz)
L=size(vel,3);
Fo=Fs/L;
freq=Fo*(0:(L-1));

[Bh,Ah] = butter(3,10/(Fs/2),'high');
[Bl,Al] = butter(3,800/(Fs/2),'low');

for ll = 1: size(vel,1)
    for mm = 1:size(vel,2)
        for nn = 1:size(vel,4)
            vel_filtH(ll,mm,:,nn)=filter(Bh,Ah,vel(ll,mm,:,nn));
            vel_filtL(ll,mm,:,nn)=filter(Bl,Al,vel_filtH(ll,mm,:,nn));
        end
    end
end

%% Make particle velocity movie
vel_filtL=vel;
if exist('Velocity_movie.avi','file')
    h = msgbox('Finished computing tissue velocity data.', 'DONE!');
else

    SAVE_MOVIE=1;

    for kk=floor(size(vel_filtL,4)/2)

        % Display movie of particle velocity image as a function of time
        if SAVE_MOVIE
            vidObj=VideoWriter(strcat('Velocity_movie', num2str(kk),'.avi'));
            open(vidObj);
            set(gcf,'nextplot','replacechildren');
        end

        for lll=1:5:size(vel_filtL,3)
            if mod(lll,2) == 0
                pause(0.2);
                figure(2);imagesc(axisWidth,axisDepth,squeeze(vel_filtL(:,:,lll)));
                hold on
                set(gcf, 'Position', [500, 500, 1000, 500]);
                clim([-3 3]);colorbar; colormap jet; grid on
                h = colorbar; ylabel(h, 'Particle velocity (mm/s)','FontSize',18,'FontWeight','bold')
                slowtime = taxis(lll);
                plottitle=['Push location at ', num2str(round(SWIFocusX_mm,2)),' mm, Particle velocity (mm/s) at:', num2str(slowtime),'s'];
                title(plottitle); axis image;
                xlabel('Lateral distance from center (mm)');
                ylabel('Axial depth (mm)');
                set(gca,'FontSize',18,'FontWeight','bold','LineWidth',2,'FontName','Arial');
                if SAVE_MOVIE
                    currFrame=getframe(gcf);
                    writeVideo(vidObj,currFrame);
                end
            end
        end
    end
    close(vidObj);
    delete(vidObj);
end
hfig=figure(2);
close(hfig);

h = msgbox('Finished computing tissue velocity data.', 'DONE!');
%%
% ---- Save IQ data (separate MAT file) ----
% Pick what you want to save:
%   IQData   = [nRows, nCols, na, nFocusX]  (your reshaped 4D IQ)
%   IQDataS  = [nRows, nCols, na]           (your squeezed IQ over acquisitions)
save('IQ_data.mat', 'IQData', '-v7.3');

% ---- Save velocity data (separate MAT file) ----
% vel is typically [ax, lat, time]
save('velocity_data.mat', 'vel', '-v7.3');

% clc;clear all; close all;