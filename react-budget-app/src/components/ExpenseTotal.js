import React, { useContext } from 'react'
import { AppContext } from '../contexts/AppContext'
import { formatNumberToWon } from '../utils'

const ExpenseTotal = () => {
  
  const {expenses} = useContext(AppContext)  
  const total = expenses.reduce((total,item)=> {
      return (total += item.cost);
  },0);

  return (
    <div className = 'alert alert-secondary p-4'>
      <span>총합계: {formatNumberToWon(total)}</span>
    </div>
  )
}

export default ExpenseTotal