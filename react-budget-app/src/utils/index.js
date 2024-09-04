export const formatNumberToWon=(number)=> {
    return `${new Intl.NumberFormat('ko-KR').format(number)}원`
}