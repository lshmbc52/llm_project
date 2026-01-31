#!/bin/bash
# 1. 변경된 모든 파일 추가
git add .
# 2. 현재 시간으로 커밋 메시지 자동 생성
git commit -m "Daily update: $(date +'%Y-%m-%d %H:%M')"
# 3. 서버로 전송
git push origin main
echo "🚀 GitHub 업로드 완료!"
