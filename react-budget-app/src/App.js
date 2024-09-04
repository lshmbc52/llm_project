
import './App.css';
import Budget from './components/Budget';
import Remaining from './components/Remaining';
import ExpenseTotal from './components/ExpenseTotal';
import ExpenseList from './components/ExpenseList';
import AddExpenseForm from './components/AddExpenseForm';


function App() {
  return (
    <div className='container'>
      <h1 className='mt-3'>지출계획</h1>
      <div className='row mt-3'>
        <div className='col-sm'>
          <Budget/>
        </div>
        <div className='col-sm'> 
          <Remaining/>
        </div>
        <div className='col-sm'>
          <ExpenseTotal/>
        </div>
      </div>

      <h3 className='mt-3'>지출들</h3>
      <div className='row'>
        <div className='col-sm'>
          <ExpenseList />
        </div>

      </div>
    
      <h3 className='mt-3'>지출추가</h3>
      <div className='row mt-3'>
        <div className='col-sm'>
          <AddExpenseForm />

        </div>
      </div>

    </div>
    
  );
}

export default App;


