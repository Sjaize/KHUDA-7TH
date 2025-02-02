## ✅ KHUDA Deep Dive 7TH 리뷰 작성 & PR 방법  

### 1️⃣ **Fork & Clone**  
```sh
git clone https://github.com/본인아이디/KHUDA-7TH.git  
cd KHUDA-7TH  
git remote add upstream https://github.com/KHUDA-7TH/KHUDA-7TH.git  
```

### 2️⃣ **새로운 브랜치 생성**  
```sh
git checkout -b review-week1  # 주차에 맞게 브랜치 생성
```

### 3️⃣ **리뷰 파일 작성**  
```sh
mkdir -p review-tasks/본인이름/week1  
cp .github/ISSUE_TEMPLATE/week-review.md review-tasks/본인이름/week1/review.md  
```
파일을 열어 내용 작성 후 저장  

### 4️⃣ **커밋 & 푸시**  
```sh
git add review-tasks/본인이름/week1/review.md  
git commit -m "Add week1 review by 본인이름"  
git push origin review-week1  
```

### 5️⃣ **GitHub에서 PR 생성**  
1. 본인 GitHub에서 "Compare & pull request" 클릭  
2. `KHUDA-7TH/KHUDA-7TH`의 `main` 브랜치로 PR 보냄  
3. "Create pull request" 클릭 🚀  

### 6️⃣ **다음 주를 위한 최신 동기화**  
```sh
git checkout main  
git pull upstream main  
git push origin main  
```
