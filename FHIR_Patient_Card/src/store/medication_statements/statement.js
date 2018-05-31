import api from '@/api.js'
import constPaths from '@/constants/constant-paths.js'

const state = {
  statements: [],
  selectedStatement: null,
  loadingStatements: true,
  totalVersions: 1,
  nextUrl: constPaths.MEDICATION_STATEMENT_URL
}

const getters = {
  statements: state => state.statements,
  loadingStatements: state => state.loadingStatements,
  totalVersions: state => state.totalVersions,
  selectedStatement: state => state.selectedStatement
}

const actions = {
  getStatements ({ state, commit }) {
    return api.fetch(state.nextUrl)
    .then(data => {
      commit('setList', data)
      return state.statements
    })
    .catch(error => Promise.reject(error))
  },
  getSingleStatement ({ state, commit }, statementID) {
    return api.fetch(constPaths.MEDICATION_STATEMENT_URL + statementID)
    .then(data => {
      commit('setSelectedStatement', data)
      return state.selectedStatement
    })
    .catch(error => Promise.reject(error))
  },
  getSingleVersionedStatement ({ state, commit }, { statementID, versionNumber }) {
    return api.fetch(constPaths.MEDICATION_STATEMENT_URL + statementID + '/_history/' + versionNumber)
    .then(data => {
      commit('setSelectedStatement', data)
      return state.selectedStatement
    })
    .catch(error => Promise.reject(error))
  },
  updateStatement ({ state, commit }, { concept, statementID, dosage }) {
    // Build changes
    var content = state.selectedStatement
    content.medicationCodeableConcept.text = concept
    content.medicationCodeableConcept.coding[0].display = concept
    content.dosage[0].text = dosage

    // Send changes
    return api.update(constPaths.MEDICATION_STATEMENT_URL + statementID, content)
    .then(data => {
      return api.fetch(constPaths.MEDICATION_STATEMENT_URL + statementID)
      .then(data => {
        commit('setSelectedStatement', data)
        return state.selectedStatement
      })
      .catch(error => Promise.reject(error))
    })
    .catch(error => Promise.reject(error))
  },
  clear ({ state, commit }) {
    commit('clear')
    return state.statements
  }
}

const mutations = {
  setList (state, data) {
    state.statements = [...state.statements, ...data.entry]
    if (data.link[1] && data.link[1].relation === 'next') {
      state.nextUrl = data.link[1].url
      state.loadingStatements = true
    } else {
      state.nextUrl = constPaths.MEDICATION_STATEMENT_URL
      state.loadingStatements = false
    }
  },
  setSelectedStatement (state, data) {
    state.selectedStatement = data
    state.totalVersions = parseInt(data.meta.versionId)
  },
  clear (state) {
    state.statements = []
    state.nextUrl = constPaths.MEDICATION_STATEMENT_URL
    state.loadingStatements = true
  }
}

export default {
  namespaced: true,
  state,
  getters,
  actions,
  mutations
}
