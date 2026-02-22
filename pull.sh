#!/bin/bash
# 1. 최신 코드 가져오기
git pull origin main

# 2. 패키지 환경 동기화
pip install -r requirements.txt

echo "✅ 코드와 환경이 모두 최신 상태로 동기화되었습니다!"
