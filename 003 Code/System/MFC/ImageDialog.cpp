// ImageDialog.cpp: 구현 파일
//

#include "pch.h"
#include "MFCApplication2.h"
#include "afxdialogex.h"
#include "ImageDialog.h"
#include "MFCApplication2Dlg.h"
#include <algorithm>
#include <gdiplus.h>

using namespace Gdiplus;
#pragma comment(lib, "gdiplus.lib")

IMPLEMENT_DYNAMIC(ImageDialog, CDialogEx)

ImageDialog::ImageDialog(CString folderPath, CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_IMAGE_DIALOG, pParent), 
	m_folderPath(folderPath), 
	m_currentResultIndex(0)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

ImageDialog::~ImageDialog()
{
	for (auto img : m_gdiplusImages)
	 	delete img;       // GDI+ Image는 delete 로 해제
	// for (auto& img : m_images)
		// img.Destroy();
}

void ImageDialog::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	// static 컨트롤
	DDX_Control(pDX, IDC_STATIC_IMG1, m_imgCtrl[0]);
	DDX_Control(pDX, IDC_STATIC_IMG2, m_imgCtrl[1]);
	DDX_Control(pDX, IDC_STATIC_IMG3, m_imgCtrl[2]);
	DDX_Control(pDX, IDC_STATIC_IMG4, m_imgCtrl[3]);

	DDX_Control(pDX, IDC_STATIC_NAME, m_staticName);
	DDX_Control(pDX, IDC_BUTTON_PREV, m_btnPrev);
	DDX_Control(pDX, IDC_BUTTON_NEXT, m_btnNext);

}

BEGIN_MESSAGE_MAP(ImageDialog, CDialogEx)
	ON_WM_PAINT()
	ON_WM_SIZE()
	ON_BN_CLICKED(IDC_BUTTON_PREV, &ImageDialog::OnBnClickedButtonPrev)
	ON_BN_CLICKED(IDC_BUTTON_NEXT, &ImageDialog::OnBnClickedButtonNext)
END_MESSAGE_MAP()

BOOL ImageDialog::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	SetIcon(m_hIcon, TRUE);
	SetIcon(m_hIcon, FALSE);

	LoadResultFolders();

	if (!m_resultFolders.empty())
		ShowResultFolder(0);
	else
		AfxMessageBox(_T("result 폴더를 찾을 수 없습니다."));

	return TRUE;
}

// --------------------------------------------------------
// prediction 폴더 내의 *_result 폴더 탐색
// --------------------------------------------------------
void ImageDialog::LoadResultFolders()
{
	CFileFind finder;
	CString searchPath = m_folderPath + _T("\\*.*");

	BOOL bWorking = finder.FindFile(searchPath);

	while (bWorking)
	{
		bWorking = finder.FindNextFile();
		if (finder.IsDots()) continue;

		if (finder.IsDirectory())
		{
			CString name = finder.GetFileName();
			if (name.Right(7) == _T("_result"))
				m_resultFolders.push_back(finder.GetFilePath());
		}
	}
	finder.Close();
	// 정렬
	std::sort(m_resultFolders.begin(), m_resultFolders.end());
}

// --------------------------------------------------------
// 해당 result 폴더의 4개 이미지를 로드 및 표시
// --------------------------------------------------------
void ImageDialog::ShowResultFolder(int index)
{
	if (index < 0 || index >= (int)m_resultFolders.size()) return;
	m_currentResultIndex = index;

	// 이전 이미지 해제
	// for (auto& img : m_images)
	// 	if (img) img.Destroy();
	// m_images.clear();
	// m_images.resize(4);
	for (auto img : m_gdiplusImages)
		delete img;
	m_gdiplusImages.assign(4, nullptr);

	CString folder = m_resultFolders[index];

	// 상단 이름 표시
	int pos = folder.ReverseFind('\\');
	CString folderName = (pos != -1) ? folder.Mid(pos + 1) : folder;
	SetDlgItemText(IDC_STATIC_NAME, folderName);
	// m_staticName.SetWindowTextW(folderName);
	
	// 하위 파일 검색
	CString foundPaths[4]; // contour, original, histogram, cell
	CFileFind finder;
	CString searchPath = folder + _T("\\*.*");
	BOOL bWorking = finder.FindFile(searchPath);

	while (bWorking)
	{
		bWorking = finder.FindNextFile();
		if (finder.IsDots() || finder.IsDirectory()) continue;

		CString filename = finder.GetFileName();
		CString full = finder.GetFilePath();

		if (filename.MakeLower().Find(_T("_original")) != -1)
			foundPaths[0] = full;
		else if (filename.MakeLower().Find(_T("_contour")) != -1)
			foundPaths[1] = full;
		else if (filename.MakeLower().Find(_T("_histogram")) != -1)
			foundPaths[2] = full;
		else if (filename.MakeLower().Find(_T("_blurrd")) != -1)
			foundPaths[3] = full;
	}

	// finder.Close();

	// 파일 로드
	for (int i = 0; i < 4; ++i)
	{
		if (!foundPaths[i].IsEmpty())
		{
			// HRESULT hr = m_images[i].Load(foundPaths[i]);
			m_gdiplusImages[i] = new Gdiplus::Image(foundPaths[i]);
			if (m_gdiplusImages[i]->GetLastStatus() != Ok)
			{
				delete m_gdiplusImages[i];
				m_gdiplusImages[i] = nullptr;
				AfxMessageBox(foundPaths[i] + _T(" 로드 실패"));
			}
		}
	}

	Invalidate();
}

// --------------------------------------------------------
// 이미지 그리기
// --------------------------------------------------------
void ImageDialog::OnPaint()
{
	CPaintDC dc(this);
	Graphics g(dc.GetSafeHdc());

	for (int i = 0; i < 4; ++i)
	{
		if (!m_gdiplusImages[i]) continue;

		CRect rect;
		m_imgCtrl[i].GetClientRect(&rect);

		Graphics imageDC(m_imgCtrl[i].GetDC()->GetSafeHdc());

		int imgW = m_gdiplusImages[i]->GetWidth();
		int imgH = m_gdiplusImages[i]->GetHeight();

		double imgAspect = (double)imgW / imgH;
		double boxAspect = (double)rect.Width() / rect.Height();

		int drawW, drawH;

		if (imgAspect > boxAspect) {
			drawW = rect.Width();
			drawH = (int)(rect.Width() / imgAspect);
		}
		else {
			drawH = rect.Height();
			drawW = (int)(rect.Height() * imgAspect);
		}

		int x = (rect.Width() - drawW) / 2;
		int y = (rect.Height() - drawH) / 2;

		imageDC.DrawImage(m_gdiplusImages[i], x, y, drawW, drawH);

		m_imgCtrl[i].ReleaseDC(m_imgCtrl[i].GetDC());
	}
	// {
	// 	if (m_images[i])
	// 	{
	// 		CDC* pDC = m_imgCtrl[i].GetDC();
	// 		if (pDC)
	// 		{
	// 			CRect rect;
	// 			m_imgCtrl[i].GetClientRect(&rect);

	// 			int imgW = m_images[i].GetWidth();
	// 			int imgH = m_images[i].GetHeight();

	// 			double imgAspect = (double)imgW / imgH;
	// 			double ctrlAspect = (double)rect.Width() / rect.Height();

	// 			int drawW, drawH;
	// 			if (imgAspect > ctrlAspect) {
	// 				drawW = rect.Width();
	// 				drawH = (int)(rect.Width() / imgAspect);
	// 			}
	// 			else {
	// 				drawH = rect.Height();
	// 				drawW = (int)(rect.Height() * imgAspect);
	// 			}

	// 			int x = (rect.Width() - drawW) / 2;
	// 			int y = (rect.Height() - drawH) / 2;

	// 			m_images[i].StretchBlt(pDC->GetSafeHdc(), x, y, drawW, drawH, SRCCOPY);
	// 			m_imgCtrl[i].ReleaseDC(pDC);
	// 		}
	// 	}
// }
}

// --------------------------------------------------------
// 이전 / 다음 폴더 이동
// --------------------------------------------------------
void ImageDialog::OnBnClickedButtonPrev()
{
	if (m_currentResultIndex > 0)
		ShowResultFolder(m_currentResultIndex - 1);
}

void ImageDialog::OnBnClickedButtonNext()
{
	if (m_currentResultIndex < (int)m_resultFolders.size() - 1)
		ShowResultFolder(m_currentResultIndex + 1);
}

// --------------------------------------------------------
// 다이얼로그　크기　조정

// --------------------------------------------------------
void ImageDialog::OnSize(UINT nType, int cx, int cy)
{
	CDialogEx::OnSize(nType, cx, cy);

	// 초기화 전 호출 방지
	if (m_imgCtrl[0].GetSafeHwnd() != nullptr)
	{
		RepositionImageControls(cx, cy);
		Invalidate();
	}
}

void ImageDialog::RepositionImageControls(int cx, int cy)
{
	if (cx <= 0 || cy <= 0) return;

	// 상단 이름 + 버튼 영역 고정
	const int topMargin = 10;
	const int nameBarTop = 15;
	const int nameBarHeight = 30;

	// 현재 컨트롤의 원래 크기를 가져와서 그대로 유지
	CRect nameRect, prevRect, nextRect;
	if (m_staticName.GetSafeHwnd())
	{
		m_staticName.GetWindowRect(&nameRect);
		ScreenToClient(&nameRect);
	}
	if (m_btnPrev.GetSafeHwnd())
	{
		m_btnPrev.GetWindowRect(&prevRect);
		ScreenToClient(&prevRect);
	}
	if (m_btnNext.GetSafeHwnd())
	{
		m_btnNext.GetWindowRect(&nextRect);
		ScreenToClient(&nextRect);
	}

	// 상단 고정된 높이 (버튼/텍스트가 포함된 구간)
	int topBarHeight = max(nameRect.bottom, nextRect.bottom) + topMargin;

	// 여백
	int marginX = 10;
	int marginY = 10;

	// 이미지 배치 가능한 영역
	int usableWidth = max(cx - marginX * 3, 100);
	int usableHeight = max(cy - topBarHeight - marginY * 3, 100);

	// 2x2 그리드로 배치
	int gridW = usableWidth / 2;
	int gridH = usableHeight / 2;

	CRect rects[4];
	rects[0] = CRect(marginX, topBarHeight + marginY, marginX + gridW, topBarHeight + marginY + gridH);
	rects[1] = CRect(marginX * 2 + gridW, topBarHeight + marginY, marginX * 2 + gridW * 2, topBarHeight + marginY + gridH);
	rects[2] = CRect(marginX, topBarHeight + marginY * 2 + gridH, marginX + gridW, topBarHeight + marginY * 2 + gridH * 2);
	rects[3] = CRect(marginX * 2 + gridW, topBarHeight + marginY * 2 + gridH, marginX * 2 + gridW * 2, topBarHeight + marginY * 2 + gridH * 2);

	for (int i = 0; i < 4; ++i)
	{
		if (m_imgCtrl[i].GetSafeHwnd())
			m_imgCtrl[i].MoveWindow(&rects[i]);
	}

	// 이름 중앙 고정 (크기 유지)
	if (m_staticName.GetSafeHwnd())
	{
		int nameWidth = nameRect.Width();
		int nameHeight = nameRect.Height();
		int nameX = (cx - nameWidth) / 2;
		int nameY = nameBarTop;
		m_staticName.MoveWindow(nameX, nameY, nameWidth, nameHeight, TRUE);

		// --- 버튼들 IDC_STATIC_NAME 오른쪽에 나란히 배치 ---
		int spacing = 5; // 버튼 사이 여백

		if (m_btnPrev.GetSafeHwnd() && m_btnNext.GetSafeHwnd())
		{
			int btnWidthPrev = prevRect.Width();
			int btnHeightPrev = prevRect.Height();
			int btnWidthNext = nextRect.Width();
			int btnHeightNext = nextRect.Height();

			int btnY = nameBarTop;

			int btnPrevX = nameX + nameWidth + spacing;
			int btnNextX = btnPrevX + btnWidthPrev + spacing;

			m_btnPrev.MoveWindow(btnPrevX, btnY, btnWidthPrev, btnHeightPrev, TRUE);
			m_btnNext.MoveWindow(btnNextX, btnY, btnWidthNext, btnHeightNext, TRUE);
		}
	}
}
